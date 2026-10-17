import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="GEX Tool", layout="wide")
R = 0.04


def bs_gamma(S, K, iv, T):
    d1 = (np.log(S / K) + (R + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    return np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) / (S * iv * np.sqrt(T))


def year_frac(exp_str):
    exp = pd.Timestamp(exp_str).tz_localize("America/New_York") + pd.Timedelta(hours=16)
    now = pd.Timestamp.now(tz="America/New_York")
    secs = (exp - now).total_seconds()
    return max(secs / (365 * 24 * 3600), 1 / (365 * 24))


@st.cache_data(ttl=300)
def get_exps(sym):
    return list(yf.Ticker(sym).options)


@st.cache_data(ttl=300)
def load_chain(sym, exp):
    t = yf.Ticker(sym)
    spot = float(t.fast_info["last_price"])
    ch = t.option_chain(exp)
    cols = ["strike", "openInterest", "impliedVolatility"]
    return spot, ch.calls[cols].copy(), ch.puts[cols].copy()


@st.cache_data(ttl=300)
def load_hist(sym, interval, period):
    return yf.Ticker(sym).history(interval=interval, period=period)


def clean(df):
    df = df.dropna()
    return df[(df.openInterest > 0) & (df.impliedVolatility > 0.05) & (df.impliedVolatility < 3)]


def net_gex_at(S, calls, puts, T):
    cg = bs_gamma(S, calls.strike.values, calls.impliedVolatility.values, T) * calls.openInterest.values
    pg = bs_gamma(S, puts.strike.values, puts.impliedVolatility.values, T) * puts.openInterest.values
    return (cg.sum() - pg.sum()) * 100 * S * S * 0.01 / 1e6


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def adx(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


st.title("أداة الجاما والسيولة")
c1, c2, c3 = st.columns(3)
sym = c1.text_input("الرمز", "TSLA").upper().strip()
rng = c3.slider("نطاق السترايكات حول السعر (%)", 5, 40, 15)

try:
    exps = get_exps(sym)
    exp = c2.selectbox("تاريخ الانتهاء", exps)
    spot, calls, puts = load_chain(sym, exp)
except Exception as e:
    st.error(f"تعذر جلب البيانات: {e}")
    st.stop()

T = year_frac(exp)
calls, puts = clean(calls), clean(puts)

df = pd.DataFrame({"strike": sorted(set(calls.strike) | set(puts.strike))})
for name, side in (("call", calls), ("put", puts)):
    g = bs_gamma(spot, side.strike.values, side.impliedVolatility.values, T) * side.openInterest.values
    g = g * 100 * spot * spot * 0.01 / 1e6
    df[name] = df.strike.map(dict(zip(side.strike, g))).fillna(0)
df["put"] = -df["put"]
df["net"] = df.call + df.put

net = df.net.sum()
grid = np.linspace(spot * 0.8, spot * 1.2, 81)
vals = np.array([net_gex_at(s, calls, puts, T) for s in grid])
cross = [grid[i] for i in range(len(grid) - 1) if vals[i] * vals[i + 1] < 0]
flip = min(cross, key=lambda x: abs(x - spot)) if cross else None

below = df[df.strike < spot]
above = df[df.strike > spot]
put_wall = below.loc[below.put.idxmin(), "strike"] if len(below) else None
call_wall = above.loc[above.call.idxmax(), "strike"] if len(above) else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("السعر", f"{spot:,.2f}")
m2.metric("الجاما الصافية ($M لكل 1%)", f"{net:,.1f}")
m3.metric("نقطة الانقلاب", f"{flip:,.1f}" if flip else "غير ظاهرة")
m4.metric("جدار الدعم / المقاومة", f"{put_wall} / {call_wall}")

view = df[(df.strike > spot * (1 - rng / 100)) & (df.strike < spot * (1 + rng / 100))]
fig = go.Figure()
fig.add_bar(y=view.strike, x=view.call, orientation="h", name="Call", marker_color="#2e9d5b")
fig.add_bar(y=view.strike, x=view.put, orientation="h", name="Put", marker_color="#d64550")
fig.add_hline(y=spot, line_dash="dash", annotation_text=f"السعر {spot:,.2f}")
if flip:
    fig.add_hline(y=flip, line_dash="dot", line_color="orange", annotation_text="الانقلاب")
fig.update_layout(barmode="relative", height=700, xaxis_title="جاما ($M لكل 1%)", yaxis_title="Strike")
st.plotly_chart(fig, use_container_width=True)

st.subheader("الإشارات الفنية")
rows = []
for label, itv, per in (("5 دقائق", "5m", "5d"), ("15 دقيقة", "15m", "5d"), ("30 دقيقة", "30m", "1mo"), ("يومي", "1d", "6mo")):
    try:
        h = load_hist(sym, itv, per).dropna()
        a, p, m = adx(h)
        r = rsi(h["Close"])
        side = "CALL" if p.iloc[-1] > m.iloc[-1] else "PUT"
        power = "قوي" if a.iloc[-1] > 25 else "ضعيف"
        rows.append({"الإطار": label, "RSI": round(r.iloc[-1], 1), "ADX": round(a.iloc[-1], 1), "الميل": f"{side} {power}"})
    except Exception:
        rows.append({"الإطار": label, "RSI": None, "ADX": None, "الميل": "لا بيانات"})
st.table(pd.DataFrame(rows))

st.subheader("القراءة")
pos = "سالبة (حركة أعنف)" if net < 0 else "موجبة (حركة أهدأ)"
lines = [f"الجاما الصافية {pos}."]
if flip:
    lines.append("السعر " + ("فوق" if spot > flip else "تحت") + f" نقطة الانقلاب {flip:,.1f}.")
if put_wall:
    lines.append(f"أقوى جدار بوت تحت السعر: {put_wall}.")
if call_wall:
    lines.append(f"أقوى جدار كول فوق السعر: {call_wall}.")
st.write("\n\n".join(lines))
st.caption("بيانات Yahoo متأخرة والـ OI يتحدث يومياً. للتعليم فقط وليست توصية استثمارية.")
