import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import akshare as ak
import re

# === 页面全局设置 ===
st.set_page_config(page_title="智能财报审计系统 (完全体)", layout="wide", initial_sidebar_state="expanded")
st.title("📊 智能财报审计系统 (龙头透视+效率分析)")


# === 核心处理引擎 (ETL) ===

def smart_load(file):
    """智能ETL函数"""
    if file is None: return None
    try:
        file.seek(0)
        try:
            df = pd.read_excel(file, header=None, engine='openpyxl')
        except:
            file.seek(0)
            df = pd.read_excel(file, header=None, engine='xlrd')

        df = df.astype(str)
        header_idx = -1
        for i in range(min(20, len(df))):
            row_str = "".join(df.iloc[i].tolist())
            if "营业收入" in row_str or "资产总计" in row_str or "经营活动" in row_str or "科目" in row_str:
                header_idx = i
                break

        if header_idx == -1: return None

        df.columns = df.iloc[header_idx]
        df = df.iloc[header_idx + 1:]
        df.columns = df.columns.str.strip().str.replace('\n', '')
        df = df.set_index(df.columns[0]).T
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
        except:
            pass
        df = df[df.index.notnull()].sort_index(ascending=False)
        for col in df.columns:
            s = df[col].astype(str).str.strip().str.replace(',', '').str.replace('--', '0')
            s = s.str.replace('nan', '0', case=False).str.replace('None', '0')
            df[col] = pd.to_numeric(s, errors='coerce').fillna(0)
        return df
    except:
        return None


def identify_table_type(df):
    if df is None: return None
    cols = "".join(df.columns.astype(str).tolist())
    if "经营活动" in cols and "现金" in cols:
        return 'csh'
    elif "资产总计" in cols or "负债合计" in cols:
        return 'bal'
    elif "营业收入" in cols and "利润" in cols:
        return 'inc'
    return None


def get_col_smart(df, keywords_list):
    for col in df.columns:
        for k in keywords_list:
            if k in str(col): return df[col], col
    return pd.Series(0, index=df.index), "未找到"


# === 辅助：获取带后缀的代码 (用于链接) ===
def get_suffix_code(code):
    c = str(code).strip()
    if c.startswith('6'): return f"sh{c}"
    if c.startswith('0') or c.startswith('3'): return f"sz{c}"
    if c.startswith('8') or c.startswith('4'): return f"bj{c}"
    return c


# === 联网获取核心信息 (含排名) ===
@st.cache_data(ttl=3600)
def get_stock_profile_full(code):
    """获取：基础信息 + 行业排名 + 龙头"""
    try:
        # 1. 基础信息
        df_info = ak.stock_individual_info_em(symbol=code)
        info_dict = dict(zip(df_info['item'], df_info['value']))
        name = info_dict.get('股票简称', '未知')
        industry = info_dict.get('行业', '未知')
        market_cap = info_dict.get('总市值', 0)

        rank_msg = "暂无数据"
        leader_msg = "暂无数据"
        tags = []

        # 2. 行业排名逻辑 (回归！)
        if industry != '未知':
            try:
                # 尝试获取行业成分股
                df_ind = ak.stock_board_industry_cons_em(symbol=industry)
                if not df_ind.empty and '总市值' in df_ind.columns:
                    # 清洗数据
                    df_ind['代码'] = df_ind['代码'].astype(str).str.strip()
                    df_ind['总市值'] = pd.to_numeric(df_ind['总市值'], errors='coerce')
                    df_ind = df_ind.sort_values('总市值', ascending=False).reset_index(drop=True)

                    # 找龙头
                    top = df_ind.iloc[0]
                    leader_msg = f"{top['名称']} ({top['代码']}) - {top['总市值'] / 1e8:.0f}亿"

                    # 找自己
                    target = df_ind[df_ind['代码'] == str(code).strip()]
                    if not target.empty:
                        rank = target.index[0] + 1
                        total = len(df_ind)
                        rank_msg = f"第 {rank} 名 / 共 {total} 家"

                        # 打标签
                        if rank == 1:
                            tags.append("👑 行业一哥")
                        elif rank <= 3:
                            tags.append("💎 行业前三")
                        elif rank <= total * 0.1:
                            tags.append("🔥 头部企业")

            except:
                pass

        # 市值标签
        mcap_b = market_cap / 1e8
        if mcap_b > 1000:
            tags.append("🐋 千亿巨头")
        elif mcap_b < 50:
            tags.append("🐟 小盘股")

        return name, industry, market_cap, rank_msg, leader_msg, tags
    except:
        return None, None, 0, "未知", "未知", []


# === 侧边栏 ===
st.sidebar.header("📁 智能投递口")
uploaded_files = st.sidebar.file_uploader("拖入文件 (利润/资产/现金)", type=['xlsx', 'xls'], accept_multiple_files=True)
years_lookback = st.sidebar.slider("审计周期", 3, 10, 5)

# === 分拣 ===
inc, bal, csh = None, None, None
detected_code = None
if uploaded_files:
    st.sidebar.markdown("---")
    for f in uploaded_files:
        if not detected_code:
            match = re.search(r'(\d{6})', f.name)
            if match: detected_code = match.group(1)
        df_t = smart_load(f)
        t_type = identify_table_type(df_t)
        if t_type == 'inc':
            inc = df_t; st.sidebar.success(f"利润: {f.name}")
        elif t_type == 'bal':
            bal = df_t; st.sidebar.success(f"资产: {f.name}")
        elif t_type == 'csh':
            csh = df_t; st.sidebar.success(f"现金: {f.name}")

# === 主程序 ===
if inc is not None and bal is not None and csh is not None:

    # --- 1. 头部：全景看板 (龙头+概念+行情) ---
    if detected_code:
        with st.spinner(f"正在全网比对 [{detected_code}] 行业地位..."):
            name, ind, cap, rank, leader, tags = get_stock_profile_full(detected_code)

        if name:
            st.markdown(f"### 🏭 {name} ({detected_code}) 深度审计报告")

            # 第一行：基本面与地位
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("所属行业", ind, f"市值 {cap / 1e8:.0f}亿")
            m2.metric("行业排名", rank, "按市值")
            m3.metric("行业龙头", leader.split(' ')[0], leader.split(' ')[-1] if '-' in leader else "")
            m4.metric("身份标签", tags[0] if tags else "无", tags[1] if len(tags) > 1 else None)

            # 第二行：传送门 (新增实时行情直达)
            st.markdown("**🔗 核心情报直达**")
            l1, l2, l3, l4 = st.columns(4)

            full_code = get_suffix_code(detected_code)  # sh603993
            em_code = full_code.upper()  # SH603993 (用于F10)

            # 按钮组
            l1.link_button("📈 实时行情 (东财)", f"https://quote.eastmoney.com/{full_code}.html")
            l2.link_button("🧩 核心题材 (F10)",
                           f"https://emweb.securities.eastmoney.com/pc_usf10/CoreConception/index?type=web&code={em_code}")
            l3.link_button("💰 行业资金流向", f"https://so.eastmoney.com/web/s?keyword={ind}资金流")
            l4.link_button("🗣️ 股吧热度", f"https://guba.eastmoney.com/list,{detected_code}.html")

            st.divider()

    # --- 数据对齐 ---
    common = inc.index.intersection(bal.index).intersection(csh.index)
    if len(common) == 0: st.error("❌ 日期无法对齐"); st.stop()
    dates = [d for d in common if d.month == 12][:years_lookback]
    if not dates: dates = common[:years_lookback]
    latest = dates[0]

    i_sub, b_sub, c_sub = inc.loc[dates], bal.loc[dates], csh.loc[dates]

    # --- 指标计算 ---
    rev, _ = get_col_smart(i_sub, ['营业总收入', '营业收入'])
    op_prof, _ = get_col_smart(i_sub, ['营业利润'])
    core_profit = op_prof - (sum([get_col_smart(i_sub, [k])[0] for k in ['公允', '投资收益', '其他收益']]))

    loss_asset, _ = get_col_smart(i_sub, ['资产减值'])
    loss_credit, _ = get_col_smart(i_sub, ['信用减值'])
    total_loss = loss_asset + loss_credit

    ocf, _ = get_col_smart(c_sub, ['经营活动产生的现金流量净额'])
    div, _ = get_col_smart(c_sub, ['分配股利', '分红'])
    capex, _ = get_col_smart(c_sub, ['购建固定'])
    repay, _ = get_col_smart(c_sub, ['偿还债务'])

    tot_asset, _ = get_col_smart(b_sub, ['资产总计'])
    op_keys = ['货币', '应收', '预付', '存货', '合同资产', '固定资产', '在建', '无形', '使用权']
    op_val = sum([get_col_smart(b_sub, [k])[0] for k in op_keys])
    non_op_val = sum([get_col_smart(b_sub, [k])[0] for k in ['交易性', '衍生', '债权', '长期股权', '投资性', '商誉']])

    op_ratio = op_val[latest] / tot_asset[latest] if tot_asset[latest] > 0 else 0
    cash_ratio = ocf[latest] / (rev[latest] + 1)

    # --- 模块展示 ---

    # 1. 利润
    st.markdown("### 1. 盈利质量 (Benefit)")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(x=dates, y=rev, title="营收规模").update_traces(marker_color='#95A5A6'),
                    use_container_width=True)
    fig2 = go.Figure(data=[
        go.Bar(name='核心主营', x=dates, y=core_profit, marker_color='#27AE60'),
        go.Bar(name='非经常性', x=dates, y=op_prof - core_profit, marker_color='#F1C40F'),
        go.Bar(name='减值', x=dates, y=total_loss, marker_color='#C0392B')
    ]).update_layout(barmode='relative', title="利润拆解")
    c2.plotly_chart(fig2, use_container_width=True)

    # 2. 资产 (新增：效率分析图)
    st.markdown("---")
    st.markdown("### 2. 资产结构与资金效率 (Debt/Assets)")

    # 左：结构图
    c3, c4 = st.columns(2)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=dates, y=op_val, stackgroup='one', name='经营资产(投入)', line_color='#2980B9'))
    fig3.add_trace(go.Scatter(x=dates, y=non_op_val, stackgroup='one', name='非经营资产', line_color='#8E44AD'))
    fig3.update_layout(title="资产属性演变")
    c3.plotly_chart(fig3, use_container_width=True)

    # 右：效率分析 (新增可视化的投入产出对比)
    # 计算投入产出比
    roi_series = (core_profit / op_val) * 100

    fig_efficiency = go.Figure()
    fig_efficiency.add_trace(go.Bar(name='经营资产投入', x=dates, y=op_val, marker_color='#2980B9', yaxis='y'))
    fig_efficiency.add_trace(
        go.Scatter(name='核心利润产出', x=dates, y=core_profit, line=dict(color='#2ECC71', width=3), yaxis='y2'))
    fig_efficiency.update_layout(
        title="<b>🚀 资金驱动效率图 (投入vs产出)</b>",
        yaxis=dict(title="资产投入 (元)", showgrid=False),
        yaxis2=dict(title="利润产出 (元)", overlaying='y', side='right', showgrid=False),
        legend=dict(x=0, y=1.1, orientation='h')
    )
    c4.plotly_chart(fig_efficiency, use_container_width=True)

    # 资金运用评价 (New!)
    op_return = core_profit[latest] / op_val[latest] if op_val[latest] > 0 else 0

    msg_capital = ""
    if op_ratio > 0.7 and op_return > 0.1:
        msg_capital = "🌟 **资金运用极度合理**：公司将绝大部分资金聚焦于主业，且产生了丰厚的回报 (ROOA > 10%)。"
    elif op_ratio > 0.7 and op_return < 0.05:
        msg_capital = "⚠️ **资金效率低下**：虽然资金都投在主业上，但产出微薄，可能处于价格战或产能过剩状态。"
    elif op_ratio < 0.5:
        msg_capital = "💣 **脱实向虚**：大量资金被挪用于理财或投资，主业资产占比过低，需警惕空心化风险。"
    else:
        msg_capital = "⚖️ **资金运用中规中矩**：资产配置均衡，效率处于正常区间。"

    st.info(msg_capital)

    # 3. 现金
    st.markdown("---")
    st.markdown("### 3. 现金流向 (Cash)")
    c5, c6 = st.columns(2)
    fig6 = go.Figure(data=[
        go.Bar(name='扩产', x=dates, y=capex, marker_color='#1ABC9C'),
        go.Bar(name='还债', x=dates, y=repay, marker_color='#95A5A6'),
        go.Bar(name='分红', x=dates, y=div, marker_color='#9B59B6')
    ]).update_layout(barmode='stack', title="资金流出去向")
    c5.plotly_chart(fig6, use_container_width=True)
    c6.plotly_chart(
        px.line(x=dates, y=(ocf / (rev + 1)) * 100, markers=True, title="净现比(%)").add_hline(y=100, line_dash="dash",
                                                                                               line_color="green"),
        use_container_width=True)

    # --- 结论 ---
    st.markdown("---")
    st.header("📝 最终结论")

    score = 60 + (15 if cash_ratio > 1 else -10) + (15 if core_profit[latest] / op_prof[latest] > 0.8 else -10) + (
        10 if div[latest] > 0 else 0)
    score = min(100, max(0, score))

    sc, txt = st.columns([1, 4])
    color = "green" if score >= 80 else "red"
    sc.markdown(f"<h1 style='color:{color};text-align:center'>{score}分</h1>", unsafe_allow_html=True)

    highlights = []
    if rank != "暂无数据" and "第 1 名" in rank: highlights.append("👑 行业绝对龙头，地位稳固")
    if op_return > 0.15: highlights.append(f"💰 赚钱机器：经营资产回报率高达 {op_return * 100:.1f}%")
    if cash_ratio > 1: highlights.append("💵 现金奶牛：回款能力极强")

    if highlights:
        for h in highlights: st.success(h)
    else:
        st.warning("暂无显著亮点，建议结合概念热度操作。")

elif uploaded_files:
    st.info("👈 文件解析中...")
else:
    st.info("👋 请在左侧上传财报文件")