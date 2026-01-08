import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import akshare as ak
import requests
import re

# === 页面全局设置 ===
st.set_page_config(page_title="智能财报审计系统 (终极完全体)", layout="wide", initial_sidebar_state="expanded")
st.title("📊 智能财报审计系统 (行业地位+题材热度+深度内功)")


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


# === 联网数据引擎 (超级缝合版) ===

def get_suffix_code(code):
    """处理代码后缀，适配不同接口"""
    c = str(code).strip()
    if c.startswith('6'): return f"SH{c}"
    if c.startswith('0') or c.startswith('3'): return f"SZ{c}"
    if c.startswith('8') or c.startswith('4'): return f"BJ{c}"
    return c


@st.cache_data(ttl=600)
def get_stock_comprehensive_info(code):
    """
    一次性获取：
    1. 基础信息 (行业、市值)
    2. 实时行情 (换手率、价格 -> 用于热度仪表盘)
    3. 行业地位 (排名、龙头 -> 用于主业护城河)
    4. 核心题材 (F10数据 -> 用于直接展示)
    """
    try:
        # 1. 基础信息
        df_info = ak.stock_individual_info_em(symbol=code)
        info_dict = dict(zip(df_info['item'], df_info['value']))
        name = info_dict.get('股票简称', '未知')
        industry = info_dict.get('行业', '未知')
        market_cap = info_dict.get('总市值', 0)

        # 2. 实时行情 (换手率)
        turnover = 0.0
        price = 0.0
        try:
            df_quote = ak.stock_zh_a_spot_em()
            target = df_quote[df_quote['代码'] == code]
            if not target.empty:
                turnover = float(target.iloc[0]['换手率'])
                price = float(target.iloc[0]['最新价'])
        except:
            pass

        # 3. 行业排名与龙头
        rank_msg = "暂无数据"
        leader_msg = "暂无数据"
        rank_int = 9999
        total_int = 1

        if industry != '未知':
            try:
                df_ind = ak.stock_board_industry_cons_em(symbol=industry)
                if not df_ind.empty and '总市值' in df_ind.columns:
                    df_ind['代码'] = df_ind['代码'].astype(str).str.strip()
                    df_ind['总市值'] = pd.to_numeric(df_ind['总市值'], errors='coerce')
                    df_ind = df_ind.sort_values('总市值', ascending=False).reset_index(drop=True)

                    top = df_ind.iloc[0]
                    leader_msg = f"{top['名称']} ({top['代码']}) - {top['总市值'] / 1e8:.0f}亿"

                    target_ind = df_ind[df_ind['代码'] == str(code).strip()]
                    total_int = len(df_ind)
                    if not target_ind.empty:
                        rank_int = target_ind.index[0] + 1
                        rank_msg = f"第 {rank_int} 名 / 共 {total_int} 家"
            except:
                pass

        # 4. 核心题材 (抓取东财F10 API)
        core_concepts = []
        try:
            suffix_code = get_suffix_code(code)
            # 这是一个公开的F10接口 URL
            url = f"https://datacenter.eastmoney.com/securities/api/data/v1/get?reportName=RPT_F10_CORE_THEME&columns=CORE_THEME&filter=(SECUCODE=%22{suffix_code.replace('SH', '.SH').replace('SZ', '.SZ')}%22)"
            res = requests.get(url, timeout=3).json()
            if res['result'] and res['result']['data']:
                # 解析一段长文本
                theme_text = res['result']['data'][0]['CORE_THEME']
                # 通常格式是 "1、概念A；2、概念B..." 或者直接一段话
                # 我们简单提取几个关键词
                parts = re.split(r'[；;、\s]', theme_text)
                # 过滤掉空字符串和数字索引
                clean_concepts = [p for p in parts if len(p) > 1 and not p.isdigit()][:3]
                core_concepts = clean_concepts
        except:
            pass

        return {
            "name": name, "industry": industry, "mcap": market_cap,
            "turnover": turnover, "price": price,
            "rank_msg": rank_msg, "leader_msg": leader_msg,
            "rank_int": rank_int, "total_int": total_int,
            "concepts": core_concepts
        }
    except Exception as e:
        return None


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

    # --- 1. 头部：终极看板 (行业地位 + 题材热度 + 核心概念) ---
    if detected_code:
        with st.spinner(f"正在全网扫描 [{detected_code}] 核心情报..."):
            info = get_stock_comprehensive_info(detected_code)

        if info:
            name = info['name']
            st.markdown(f"### 🏭 {name} ({detected_code}) 深度透视看板")

            # === A. 双仪表盘逻辑 ===
            # 1. 主业护城河 (Industry Score)
            # 逻辑：市值越大 + 排名越靠前 = 分数越高
            mcap_score = min(60, int((info['mcap'] / 100000000000) * 60))  # 千亿市值拿60分
            rank_score = 0
            if info['rank_int'] == 1:
                rank_score = 40
            elif info['rank_int'] <= 5:
                rank_score = 30
            elif info['rank_int'] <= 20:
                rank_score = 20
            else:
                rank_score = 10
            industry_moat = min(100, mcap_score + rank_score)

            # 2. 题材关注度 (Concept Heat)
            # 逻辑：基于换手率。>15%极热，>7%热，>3%温，<1%冷
            turnover = info['turnover']
            concept_heat = min(100, int((turnover / 15.0) * 100))

            # === B. 布局展示 ===
            # 第一排：三个专栏
            col_ind, col_con, col_tags = st.columns([1.5, 1.5, 1.2])

            with col_ind:
                st.markdown(f"**🔵 主业护城河 (行业地位)**")
                st.progress(industry_moat)
                c1, c2 = st.columns(2)
                c1.metric("所属行业", info['industry'])
                c2.metric("行业排名", info['rank_int'], f"共{info['total_int']}家")
                st.caption(f"行业龙头: {info['leader_msg']}")

            with col_con:
                st.markdown(f"**🔴 题材关注度 (资金热度)**")
                st.progress(concept_heat)
                c3, c4 = st.columns(2)
                c3.metric("实时换手", f"{turnover}%")
                heat_label = "🔥 极热" if turnover > 10 else "📈 活跃" if turnover > 5 else "❄️ 冷门"
                c4.metric("热度评级", heat_label)
                st.caption(f"当前股价: {info['price']} 元")

            with col_tags:
                st.markdown("**🧩 核心概念 (Direct)**")
                if info['concepts']:
                    # 直接显示标签，不再只是链接
                    for tag in info['concepts']:
                        st.markdown(f"#### `🏷️ {tag}`")
                else:
                    st.info("暂未提取到核心题材")

            # 第二排：传送门按钮
            st.markdown("---")
            b1, b2, b3, b4 = st.columns(4)
            full_code = get_suffix_code(detected_code)

            b1.link_button("📈 实时行情直达", f"https://quote.eastmoney.com/{full_code.lower()}.html")
            b2.link_button("🧩 更多题材 (F10)",
                           f"https://emweb.securities.eastmoney.com/pc_usf10/CoreConception/index?type=web&code={full_code.upper()}")
            b3.link_button("💰 行业资金流向", f"https://so.eastmoney.com/web/s?keyword={info['industry']}资金流")
            b4.link_button("🗣️ 股吧讨论热度", f"https://guba.eastmoney.com/list,{detected_code}.html")

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

    # 2. 资产 (保留效率图)
    st.markdown("---")
    st.markdown("### 2. 资产结构与资金效率 (Debt/Assets)")

    c3, c4 = st.columns(2)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=dates, y=op_val, stackgroup='one', name='经营资产(投入)', line_color='#2980B9'))
    fig3.add_trace(go.Scatter(x=dates, y=non_op_val, stackgroup='one', name='非经营资产', line_color='#8E44AD'))
    fig3.update_layout(title="资产属性演变")
    c3.plotly_chart(fig3, use_container_width=True)

    # 效率分析图 (双轴)
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

    # 资金运用智能点评
    op_return = core_profit[latest] / op_val[latest] if op_val[latest] > 0 else 0
    msg_capital = ""
    if op_ratio > 0.7 and op_return > 0.1:
        msg_capital = f"🌟 **资金运用极度合理**：公司将 {op_ratio * 100:.0f}% 的资金聚焦于主业，且每一分钱投入都创造了丰厚的回报 (回报率 {op_return * 100:.1f}%)。"
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
    # 行业地位亮点
    if info['rank_int'] == 1:
        highlights.append(f"👑 行业绝对龙头 (排名第1)")
    elif info['rank_int'] <= 5:
        highlights.append(f"💎 行业头部企业 (排名第{info['rank_int']})")

    # 财务亮点
    if op_return > 0.15: highlights.append(f"💰 赚钱机器：经营资产回报率高达 {op_return * 100:.1f}%")
    if cash_ratio > 1: highlights.append("💵 现金奶牛：回款能力极强")

    if highlights:
        for h in highlights: st.success(h)
    else:
        st.warning("暂无显著财务亮点，建议关注题材热度。")

elif uploaded_files:
    st.info("👈 文件解析中...")
else:
    st.info("👋 请在左侧上传财报文件")