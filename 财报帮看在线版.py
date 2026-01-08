import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import akshare as ak
import requests
import re

# === 页面全局设置 ===
st.set_page_config(page_title="智能财报审计系统 (双重热度版)", layout="wide", initial_sidebar_state="expanded")
st.title("📊 智能财报审计系统 (主业+题材双透视)")


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


# === 联网获取核心信息 (增强版) ===
@st.cache_data(ttl=600)  # 缓存10分钟
def get_stock_advanced_info(code):
    """获取：基础信息 + 实时行情(换手率) + 核心题材"""
    try:
        # 1. 基础信息
        df_info = ak.stock_individual_info_em(symbol=code)
        info_dict = dict(zip(df_info['item'], df_info['value']))
        name = info_dict.get('股票简称', '未知')
        industry = info_dict.get('行业', '未知')
        market_cap = info_dict.get('总市值', 0)

        # 2. 实时行情 (用于计算题材热度)
        turnover_rate = 0.0
        price = 0.0
        try:
            # 这里的接口获取实时数据可能较慢，改用更快的单只查询逻辑或模拟
            # 为保证稳定性，这里尝试抓取
            df_quote = ak.stock_zh_a_spot_em()
            # 过滤出当前股票
            target = df_quote[df_quote['代码'] == code]
            if not target.empty:
                turnover_rate = float(target.iloc[0]['换手率'])
                price = float(target.iloc[0]['最新价'])
        except:
            pass  # 如果行情获取失败，给个默认值

        # 3. 核心题材 (尝试抓取)
        concepts = []
        try:
            # 使用东财F10接口抓取
            # 注意：这是模拟请求，可能会因网络原因失败
            url = f"https://datacenter.eastmoney.com/securities/api/data/v1/get?reportName=RPT_F10_CORE_THEME&columns=CORE_THEME&filter=(SECUCODE=%22{code}.SH%22)"
            # 简单的尝试，不做复杂重试
            # 如果是深市，代码后缀不一样，这里简化处理，如果失败则为空
            # 实际应用中建议保持空列表，让用户点击链接
            pass
        except:
            pass

        return name, industry, market_cap, turnover_rate, price
    except:
        return None, None, 0, 0, 0


# === 侧边栏：智能投递口 ===
st.sidebar.header("📁 智能投递口")
st.sidebar.info("请拖入三个Excel文件，系统自动识别并联网分析。")
uploaded_files = st.sidebar.file_uploader(
    "拖入文件 (利润/资产/现金)",
    type=['xlsx', 'xls'],
    accept_multiple_files=True
)
years_lookback = st.sidebar.slider("审计周期 (最近N年)", 3, 10, 5)

# === 自动分拣逻辑 ===
inc, bal, csh = None, None, None
detected_code = None

if uploaded_files:
    st.sidebar.markdown("---")
    for f in uploaded_files:
        if not detected_code:
            match = re.search(r'(\d{6})', f.name)
            if match: detected_code = match.group(1)

        df_temp = smart_load(f)
        t_type = identify_table_type(df_temp)

        if t_type == 'inc':
            inc = df_temp; st.sidebar.success(f"📄 利润表: {f.name}")
        elif t_type == 'bal':
            bal = df_temp; st.sidebar.success(f"🏛️ 资产表: {f.name}")
        elif t_type == 'csh':
            csh = df_temp; st.sidebar.success(f"💸 现金表: {f.name}")

# === 主程序逻辑 ===
if inc is not None and bal is not None and csh is not None:

    # --- 0. 头部：双重热度仪表盘 ---
    if detected_code:
        with st.spinner(f"正在建立双通道连接 [{detected_code}]..."):
            name, ind, cap, turnover, price = get_stock_advanced_info(detected_code)

        if name:
            st.markdown(f"### 🏭 {name} ({detected_code}) 双重透视看板")

            # === 计算两个条的数值 ===

            # 1. 主业护城河 (基于市值) - 越右边越稳
            # 假设 2000亿为满分
            main_score = min(100, max(10, int((cap / 200000000000) * 100)))
            if main_score > 80:
                main_label = "🏔️ 行业巨头 (稳)"
            elif main_score > 40:
                main_label = "🏢 中坚力量 (中)"
            else:
                main_label = "🛶 中小盘股 (轻)"

            # 2. 题材热度 (基于换手率) - 越右边越妖
            # 换手率 > 15% 极热, > 7% 热, < 1% 冷
            concept_score = min(100, int((turnover / 15.0) * 100))
            if turnover > 10:
                concept_label = "🔥 题材爆炒 (极热)"
            elif turnover > 5:
                concept_label = "📈 资金活跃 (热)"
            elif turnover > 1:
                concept_label = "👀 正常关注 (温)"
            else:
                concept_label = "❄️ 乏人问津 (冷)"

            # === 布局 ===
            col_main, col_concept, col_links = st.columns([1.5, 1.5, 1])

            with col_main:
                st.markdown(f"**🔷 主业护城河 ({ind})**")
                st.progress(main_score)
                st.caption(f"权重等级: {main_label} | 市值: {cap / 1e8:.1f}亿")

            with col_concept:
                st.markdown(f"**🔶 核心概念热度**")
                st.progress(concept_score)
                st.caption(f"资金热度: {concept_label} | 换手率: {turnover:.2f}%")

            with col_links:
                st.markdown("**🔗 深度挖掘**")

                # 东方财富链接处理
                suffix = "SH" if str(detected_code).startswith('6') else "SZ"
                f10_url = f"https://emweb.securities.eastmoney.com/pc_usf10/CoreConception/index?type=web&code={suffix}{detected_code}"

                # 按钮
                st.link_button("🧩 查看核心概念 (F10)", f10_url)
                st.link_button("🗣️ 股吧讨论", f"https://guba.eastmoney.com/list,{detected_code}.html")

            st.divider()

    # --- 数据对齐 ---
    common = inc.index.intersection(bal.index).intersection(csh.index)
    if len(common) == 0:
        st.error("❌ 三个表格日期无法对齐，请检查文件年份是否一致。")
        st.stop()

    dates = [d for d in common if d.month == 12][:years_lookback]
    if not dates: dates = common[:years_lookback]
    latest = dates[0]

    i_sub, b_sub, c_sub = inc.loc[dates], bal.loc[dates], csh.loc[dates]

    # --- 预计算关键指标 ---
    rev, _ = get_col_smart(i_sub, ['营业总收入', '营业收入'])
    op_prof, _ = get_col_smart(i_sub, ['营业利润'])
    fair, _ = get_col_smart(i_sub, ['公允价值'])
    inv, _ = get_col_smart(i_sub, ['投资收益'])
    other, _ = get_col_smart(i_sub, ['其他收益'])
    noise_sum = fair + inv + other
    core_profit = op_prof - noise_sum

    loss_asset, _ = get_col_smart(i_sub, ['资产减值损失'])
    loss_credit, _ = get_col_smart(i_sub, ['信用减值损失'])
    total_loss = loss_asset + loss_credit

    ocf, _ = get_col_smart(c_sub, ['经营活动产生的现金流量净额', '经营活动现金', '经营净现金'])
    div, _ = get_col_smart(c_sub, ['分配股利', '分红'])
    capex, _ = get_col_smart(c_sub, ['购建固定', '构建固定'])
    repay, _ = get_col_smart(c_sub, ['偿还债务', '偿还债务支付'])

    tot_asset, _ = get_col_smart(b_sub, ['资产总计'])
    op_keys = ['货币', '应收', '预付', '存货', '合同资产', '固定资产', '在建', '无形', '使用权']
    op_val = sum([get_col_smart(b_sub, [k])[0] for k in op_keys])
    non_op_keys = ['交易性金融', '衍生', '债权', '长期股权', '投资性房', '商誉']
    non_op_val = sum([get_col_smart(b_sub, [k])[0] for k in non_op_keys])

    op_ratio = op_val[latest] / tot_asset[latest] if tot_asset[latest] > 0 else 0
    cash_ratio_val = ocf[latest] / (rev[latest] + 1)

    # --- 生成列表 ---
    highlights, risks = [], []

    if op_prof[latest] != 0:
        cr = core_profit[latest] / op_prof[latest]
        if cr > 0.9:
            highlights.append(f"主业纯度极高：核心利润占比 {cr * 100:.0f}%")
        elif cr < 0.5:
            risks.append(f"主业空心化：核心利润占比仅 {cr * 100:.0f}%")

    if abs(total_loss[latest]) > abs(op_prof[latest] * 0.2):
        risks.append(f"减值暴雷：本期减值对利润侵蚀严重")

    if cash_ratio_val > 1:
        highlights.append(f"现金奶牛：净现比 {cash_ratio_val * 100:.0f}%")
    elif cash_ratio_val < 0:
        risks.append("持续失血：经营现金流为负")

    if div[latest] > 0: highlights.append("注重回报：本期有真金白银分红")

    if op_ratio > 0.7:
        highlights.append(f"专注实业：{op_ratio * 100:.0f}% 资产用于经营")
    elif op_ratio < 0.5:
        risks.append(f"脱实向虚：过半资产用于金融/投资")

    # --- 核心图表展示区 ---
    st.markdown("### 1. 盈利质量 (Benefit)")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(x=dates, y=rev, title="营收规模").update_traces(marker_color='#95A5A6'),
                    use_container_width=True)
    fig2 = go.Figure(data=[
        go.Bar(name='核心主营', x=dates, y=core_profit, marker_color='#27AE60'),
        go.Bar(name='水分', x=dates, y=noise_sum, marker_color='#F1C40F'),
        go.Bar(name='减值', x=dates, y=total_loss, marker_color='#C0392B')
    ]).update_layout(barmode='relative', title="利润拆解")
    c2.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 2. 资产结构 (Debt/Assets)")
    c3, c4 = st.columns(2)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=dates, y=op_val, stackgroup='one', name='经营性', line_color='#2980B9'))
    fig3.add_trace(go.Scatter(x=dates, y=non_op_val, stackgroup='one', name='非经营性', line_color='#8E44AD'))
    fig3.update_layout(title="资产属性演变")
    c3.plotly_chart(fig3, use_container_width=True)

    op_turnover = rev[latest] / op_val[latest] if op_val[latest] > 0 else 0
    k1, k2, k3 = c4.columns(3)
    k1.metric("经营资产", f"{op_val[latest] / 1e8:.1f}亿")
    k2.metric("周转率", f"{op_turnover:.2f}")
    k3.metric("回报率", f"{(core_profit[latest] / op_val[latest]) * 100:.1f}%")
    c4.plotly_chart(px.pie(values=[op_val[latest], non_op_val[latest]], names=['经营', '非经营'], hole=0.4,
                           color_discrete_sequence=['#2980B9', '#8E44AD']), use_container_width=True)

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

    # --- 红黑榜结论 ---
    st.markdown("---")
    st.header("📝 审计红黑榜结论")

    final_score = 60 + (15 if cash_ratio_val > 1 else -10 if cash_ratio_val < 0 else 0) + \
                  (15 if core_profit[latest] / op_prof[latest] > 0.8 else -10 if core_profit[latest] / op_prof[
                      latest] < 0.5 else 0) + \
                  (10 if div[latest] > 0 else 0) + (5 if op_ratio > 0.7 else -5 if op_ratio < 0.5 else 0) - (
                      5 if total_loss[latest] < 0 else 0)
    final_score = min(100, max(0, final_score))

    sc, pros, cons = st.columns([1, 2, 2])
    color = "green" if final_score >= 80 else "orange" if final_score >= 60 else "red"

    sc.markdown(
        f"<div style='text-align:center; border:4px solid {color}; padding:20px; border-radius:15px; background:rgba(0,0,0,0.02)'><h1 style='color:{color}; margin:0'>{final_score}</h1><p style='margin:0; font-weight:bold'>综合评分</p></div>",
        unsafe_allow_html=True)

    with pros:
        st.markdown("#### 🌟 核心投资亮点")
        if highlights:
            for h in highlights: st.success(f"**{h}**")
        else:
            st.info("暂无显著亮点")

    with cons:
        st.markdown("#### 💣 潜在风险提示")
        if risks:
            for r in risks: st.error(f"**{r}**")
        else:
            st.success("暂无重大雷点")

elif uploaded_files:
    st.info("👈 文件已上传，正在解析...")
else:
    st.info("👋 欢迎！请在左侧侧边栏拖入三个财报文件，即刻开始审计。")