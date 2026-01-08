import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import akshare as ak
import re

# === 页面全局设置 ===
st.set_page_config(page_title="智能财报审计系统 (红黑榜修复版)", layout="wide", initial_sidebar_state="expanded")
st.title("📊 智能财报审计系统 (热度透视+红黑榜)")


# === 核心处理引擎 (ETL) ===

def smart_load(file):
    """智能ETL函数：读取并清洗数据"""
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


# === 联网获取基础信息 (移除排名，保留基本面) ===
@st.cache_data(ttl=3600)
def get_stock_basic(code):
    try:
        df_info = ak.stock_individual_info_em(symbol=code)
        info_dict = dict(zip(df_info['item'], df_info['value']))
        name = info_dict.get('股票简称', '未知')
        industry = info_dict.get('行业', '未知')
        market_cap = info_dict.get('总市值', 0)
        return name, industry, market_cap
    except:
        return None, None, 0


# === 主程序逻辑 ===
if inc is not None and bal is not None and csh is not None:

    # --- 0. 头部：股票画像与市场热度 ---
    if detected_code:
        with st.spinner(f"正在连接数据中心，获取 [{detected_code}] 市场情报..."):
            name, ind, cap = get_stock_basic(detected_code)

        if name:
            st.markdown(f"### 🏭 {name} ({detected_code}) 深度审计报告")

            # 计算一个“理论热度值” (基于市值的简单算法，模拟热度)
            # 千亿市值以上热度自动设为高
            heat_score = min(100, max(10, int((cap / 100000000000) * 100)))
            if heat_score < 20:
                heat_level = "❄️ 散户冷门"
            elif heat_score < 60:
                heat_level = "🔥 市场热门"
            else:
                heat_level = "🌟 全民焦点"

            # 布局：基本面 + 关注度传送门
            col_info, col_heat = st.columns([2, 1])

            with col_info:
                m1, m2, m3 = st.columns(3)
                m1.metric("所属行业", ind)
                m2.metric("总市值", f"{cap / 1e8:.1f} 亿")
                m3.metric("市场关注级", heat_level, f"热度指数 {heat_score}")
                st.progress(heat_score)

            with col_heat:
                st.markdown("**🔍 投资者情报中心 (一键直达)**")
                # 东方财富行业榜
                st.link_button("📈 东方财富-行业排行", f"https://data.eastmoney.com/bkzj/{ind}.html")

                # 百度指数 & 股吧
                c_h1, c_h2 = st.columns(2)
                c_h1.link_button("🔍 百度搜索指数",
                                 f"https://index.baidu.com/v2/main/index.html#/trend/{name}?words={name}")
                c_h2.link_button("🗣️ 股吧讨论热度", f"https://guba.eastmoney.com/list,{detected_code}.html")

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

    # --- 生成亮点与风险 (纯文本列表，防止DeltaGenerator报错) ---
    highlights, risks = [], []

    # 利润判断
    if op_prof[latest] != 0:
        cr = core_profit[latest] / op_prof[latest]
        if cr > 0.9:
            highlights.append(f"主业纯度极高：核心利润占比 {cr * 100:.0f}%")
        elif cr < 0.5:
            risks.append(f"主业空心化：核心利润占比仅 {cr * 100:.0f}%，依赖非经常性损益")

    # 减值判断
    if abs(total_loss[latest]) > abs(op_prof[latest] * 0.2):
        risks.append(f"减值暴雷：本期减值对利润侵蚀严重")

    # 现金流判断
    if cash_ratio_val > 1:
        highlights.append(f"现金奶牛：净现比 {cash_ratio_val * 100:.0f}%，利润含金量高")
    elif cash_ratio_val < 0:
        risks.append("持续失血：经营现金流为负，造血能力差")

    # 分红判断
    if div[latest] > 0: highlights.append("注重回报：本期有真金白银分红")

    # 资产结构
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

    # --- 红黑榜结论 (BUG修复版) ---
    st.markdown("---")
    st.header("📝 审计红黑榜结论")

    # 计算总分
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

    # 修复 DeltaGenerator 报错的关键：
    # 错误写法: [st.success(h) for h in highlights] -> 这会返回一个对象列表并被打印
    # 正确写法: 使用明确的 for 循环，不返回列表

    with pros:
        st.markdown("#### 🌟 核心投资亮点")
        if highlights:
            for h in highlights:
                st.success(f"**{h}**")
        else:
            st.info("暂无显著亮点")

    with cons:
        st.markdown("#### 💣 潜在风险提示")
        if risks:
            for r in risks:
                st.error(f"**{r}**")
        else:
            st.success("暂无重大雷点")

elif uploaded_files:
    st.info("👈 文件已上传，正在解析...")
else:
    st.info("👋 欢迎！请在左侧侧边栏拖入三个财报文件，即刻开始审计。")