import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import akshare as ak
import re

# === 页面全局设置 ===
st.set_page_config(page_title="智能财报审计系统 (修复加强版)", layout="wide", initial_sidebar_state="expanded")
st.title("📊 智能财报审计系统 (稳定性修复版)")

# === 侧边栏：数据导入 ===
st.sidebar.header("📁 审计底稿导入")
st.sidebar.info("文件名若包含股票代码(如603993)，系统自动透视行业地位。")
file_inc = st.sidebar.file_uploader("1. 利润表 (含营业收入/减值损失)", type=['xlsx', 'xls'])
file_bal = st.sidebar.file_uploader("2. 资产负债表 (含资产总计)", type=['xlsx', 'xls'])
file_csh = st.sidebar.file_uploader("3. 现金流量表 (含经营现金流/分红)", type=['xlsx', 'xls'])

years_lookback = st.sidebar.slider("审计周期 (最近N年)", 3, 10, 5)
show_debug = st.sidebar.checkbox("🛠️ 开启调试模式")


# === 🧠 核心升级：行业地位透视 (修复排名匹配问题) ===
@st.cache_data(ttl=3600)
def get_stock_profile_advanced(code):
    """联网获取：基本信息 + 行业排名 + 绝对龙头"""
    try:
        # 1. 获取个股基本信息
        df_info = ak.stock_individual_info_em(symbol=code)
        info_dict = dict(zip(df_info['item'], df_info['value']))

        name = info_dict.get('股票简称', '未知')
        industry = info_dict.get('行业', '未知')
        market_cap = info_dict.get('总市值', 0)

        rank_msg = "暂无数据"
        leader_msg = "暂无数据"

        # 2. 获取同行业数据并排名
        if industry != '未知':
            try:
                # 获取该行业所有股票
                df_industry = ak.stock_board_industry_cons_em(symbol=industry)

                if '总市值' in df_industry.columns:
                    # === 关键修复：强制转换代码列为字符串，去除空格 ===
                    df_industry['代码'] = df_industry['代码'].astype(str).str.strip()
                    clean_code = str(code).strip()

                    # 排序
                    df_industry['总市值'] = pd.to_numeric(df_industry['总市值'], errors='coerce')
                    df_industry = df_industry.sort_values('总市值', ascending=False).reset_index(drop=True)

                    # A. 找龙头
                    if not df_industry.empty:
                        top_stock = df_industry.iloc[0]
                        leader_name = top_stock['名称']
                        leader_code = top_stock['代码']
                        leader_mcap = top_stock['总市值'] / 100000000
                        leader_msg = f"{leader_name} ({leader_code}) - {leader_mcap:.0f}亿"

                    # B. 找排名 (使用字符串严格匹配)
                    target = df_industry[df_industry['代码'] == clean_code]
                    if not target.empty:
                        rank = target.index[0] + 1
                        total_count = len(df_industry)
                        rank_msg = f"第 {rank} 名 / 共 {total_count} 家"
            except Exception as e:
                # 某些冷门行业可能接口报错，忽略
                pass

        # 3. 标签逻辑
        tags = []
        try:
            mcap_billion = market_cap / 100000000
            if mcap_billion > 1000:
                tags.append("🔥 千亿巨头")
            elif mcap_billion > 300:
                tags.append("💎 行业龙头")
            elif mcap_billion > 100:
                tags.append("🏢 知名大票")
            else:
                tags.append("🐟 中小盘股")

            if "第 1 名" in rank_msg:
                tags.append("👑 绝对一哥")
        except:
            pass

        return name, industry, market_cap, rank_msg, leader_msg, tags

    except Exception as e:
        return None, None, None, None, None, []


# === 自动识别代码 ===
detected_code = None
uploaded_files = [f for f in [file_inc, file_bal, file_csh] if f is not None]

if uploaded_files:
    for f in uploaded_files:
        match = re.search(r'(\d{6})', f.name)
        if match:
            detected_code = match.group(1)
            break

if detected_code:
    with st.spinner(f"正在透视 [{detected_code}] 的行业地位..."):
        name, ind, cap, rank, leader, tags = get_stock_profile_advanced(detected_code)

    if name:
        st.markdown(f"### 🏭 {name} ({detected_code}) 行业地位透视")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("所属行业", ind, f"总市值 {cap / 100000000:.1f}亿")
        m2.metric("行业排名", rank, "按市值排序")
        m3.metric("行业绝对龙头", leader.split(' ')[0], leader.split(' ')[-1])
        m4.metric("企业标签", tags[0] if tags else "无", tags[1] if len(tags) > 1 else None)
        st.divider()


# === 核心处理引擎 ===

def smart_load(file):
    if file is None: return None
    try:
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


def get_col_smart(df, keywords_list):
    for col in df.columns:
        for k in keywords_list:
            if k in str(col): return df[col], col
    return pd.Series(0, index=df.index), "未找到"


def generate_comments(inc, bal, csh, dates):
    latest = dates[0]
    comments = {"good": [], "bad": [], "neutral": []}

    op_prof, _ = get_col_smart(inc, ['营业利润'])
    fair, _ = get_col_smart(inc, ['公允价值'])
    inv, _ = get_col_smart(inc, ['投资收益'])
    other, _ = get_col_smart(inc, ['其他收益'])
    core = op_prof - fair - inv - other

    if op_prof[latest] != 0:
        ratio = core[latest] / op_prof[latest]
        if ratio > 0.9:
            comments["good"].append(f"主业极强：核心利润占比 {ratio * 100:.0f}%，利润含金量极高")
        elif ratio < 0.5:
            comments["bad"].append(f"主业空心化：核心利润占比仅 {ratio * 100:.0f}%，严重依赖投资或补贴")

    loss_asset, _ = get_col_smart(inc, ['资产减值损失'])
    loss_credit, _ = get_col_smart(inc, ['信用减值损失'])
    total_loss = loss_asset + loss_credit
    if abs(total_loss[latest]) > abs(op_prof[latest] * 0.2):
        comments["bad"].append(
            f"减值雷区：本期减值损失对利润侵蚀严重 (占比>{abs(total_loss[latest] / op_prof[latest] * 100):.0f}%)")

    rev, _ = get_col_smart(inc, ['营业收入', '营业总收入'])
    ocf, _ = get_col_smart(csh, ['经营活动产生的现金流量净额', '经营活动现金', '经营净现金'])
    cash_ratio = ocf[latest] / (rev[latest] + 1)

    if cash_ratio > 1.0:
        comments["good"].append("现金奶牛：净现比 > 100%，产业链话语权强")
    elif cash_ratio < 0:
        comments["bad"].append("持续失血：经营现金流为负，造血能力堪忧")

    div, _ = get_col_smart(csh, ['分配股利', '分红'])
    if div[latest] > 0: comments["good"].append("注重回报：本期有真金白银的分红支出")
    return comments


# === 主程序 ===

if file_inc and file_bal and file_csh:
    if st.button("🚀 启动深度审计", type="primary"):
        with st.spinner("AI 审计员正在核对数据..."):
            inc = smart_load(file_inc)
            bal = smart_load(file_bal)
            csh = smart_load(file_csh)

        if inc is not None and bal is not None and csh is not None:
            # === 关键修复：检查日期交集 ===
            common = inc.index.intersection(bal.index).intersection(csh.index)

            if len(common) == 0:
                st.error("❌ 严重错误：三个表格中没有找到**共同的日期**！")
                st.warning("""
                **可能原因及建议：**
                1. **年份不匹配**：比如利润表是2024年，但资产表是2020年？
                2. **日期格式问题**：请勾选左侧侧边栏的【🛠️ 开启调试模式】，查看读取到的原始列名和索引。
                3. **文件错误**：请检查是否上传了空文件或错误的文件。
                """)
                if show_debug:
                    st.write("利润表日期:", inc.index.tolist())
                    st.write("资产表日期:", bal.index.tolist())
                    st.write("现金表日期:", csh.index.tolist())
                st.stop()  # 停止运行，防止后续崩溃

            dates = [d for d in common if d.month == 12][:years_lookback]
            if not dates: dates = common[:years_lookback]

            i_sub = inc.loc[dates]
            b_sub = bal.loc[dates]
            c_sub = csh.loc[dates]
            latest = dates[0]

            # 预计算
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

            tot_asset, _ = get_col_smart(b_sub, ['资产总计'])
            op_keys = ['货币', '应收票据', '应收账款', '预付', '存货', '合同资产', '固定资产', '在建工程', '无形资产',
                       '使用权']
            op_val = sum([get_col_smart(b_sub, [k])[0] for k in op_keys])
            non_op_keys = ['交易性金融', '衍生金融', '债权投资', '其他债权', '长期股权', '投资性房地', '商誉']
            non_op_val = sum([get_col_smart(b_sub, [k])[0] for k in non_op_keys])

            op_ratio = op_val[latest] / tot_asset[latest] if tot_asset[latest] > 0 else 0
            cash_ratio_val = ocf[latest] / (rev[latest] + 1)
            comments = generate_comments(inc, bal, csh, dates)

            # === 模块一：利润质量 ===
            st.markdown("### 1. 盈利质量与减值扰动")
            c1, c2 = st.columns(2)
            with c1:
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(x=dates, y=rev, name='营业收入', marker_color='#95A5A6'))
                fig1.add_trace(go.Bar(x=dates, y=op_prof, name='营业利润', marker_color='#34495E'))
                fig1.update_layout(title="营收 vs 营业利润")
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = go.Figure(data=[
                    go.Bar(name='核心主营利润', x=dates, y=core_profit, marker_color='#27AE60'),
                    go.Bar(name='非经常性收益', x=dates, y=noise_sum, marker_color='#F1C40F'),
                    go.Bar(name='减值损失(雷)', x=dates, y=total_loss, marker_color='#C0392B')
                ])
                fig2.update_layout(barmode='relative', title="利润深度拆解")
                st.plotly_chart(fig2, use_container_width=True)
            if comments["bad"]:
                for c in comments["bad"]: st.error(f"❌ {c}")
            if comments["good"]:
                for c in comments["good"]: st.success(f"✅ {c}")

            # === 模块二：资产结构 ===
            st.markdown("---")
            st.markdown("### 2. 资产结构与经营效率")
            c3, c4 = st.columns(2)
            with c3:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=dates, y=op_val, stackgroup='one', name='经营性资产', line_color='#2980B9'))
                fig3.add_trace(
                    go.Scatter(x=dates, y=non_op_val, stackgroup='one', name='非经营性资产', line_color='#8E44AD'))
                fig3.update_layout(title="资产属性演变")
                st.plotly_chart(fig3, use_container_width=True)
            with c4:
                labels = ['经营性资产', '非经营性资产']
                values = [op_val[latest], non_op_val[latest]]
                fig_pie = px.pie(values=values, names=labels, hole=0.4, title=f"{latest.date()} 资产配置",
                                 color_discrete_sequence=['#2980B9', '#8E44AD'])
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("#### ⚙️ 经营资产效率")
            op_turnover = rev[latest] / op_val[latest] if op_val[latest] > 0 else 0
            op_return = core_profit[latest] / op_val[latest] if op_val[latest] > 0 else 0
            k1, k2, k3 = st.columns(3)
            k1.metric("经营性资产投入", f"{op_val[latest] / 100000000:.2f} 亿")
            k2.metric("周转率 (营收/资产)", f"{op_turnover:.2f} 倍")
            k3.metric("回报率 (利润/资产)", f"{op_return * 100:.1f}%")
            if op_ratio > 0.7:
                st.success(f"✅ **专注主业**：{op_ratio * 100:.0f}% 的资金都在干正事。")
            elif op_ratio < 0.5:
                st.warning(f"⚠️ **脱实向虚**：仅 {op_ratio * 100:.0f}% 的资金在经营，需警惕。")

            # === 模块三：资金去向 ===
            st.markdown("---")
            st.markdown("### 3. 现金流透视")
            capex, n_capex = get_col_smart(c_sub, ['购建固定', '构建固定'])
            repay, n_repay = get_col_smart(c_sub, ['偿还债务', '偿还债务支付'])
            c5, c6 = st.columns(2)
            with c5:
                if capex.sum() == 0 and repay.sum() == 0 and div.sum() == 0:
                    st.warning("⚠️ 未找到现金流出明细")
                else:
                    fig6 = go.Figure(data=[
                        go.Bar(name='扩产投入', x=dates, y=capex, marker_color='#1ABC9C'),
                        go.Bar(name='偿还债务', x=dates, y=repay, marker_color='#95A5A6'),
                        go.Bar(name='分红回报', x=dates, y=div, marker_color='#9B59B6')
                    ])
                    fig6.update_layout(barmode='stack', title="资金流出结构")
                    st.plotly_chart(fig6, use_container_width=True)
            with c6:
                cash_ratio_pct = (ocf / (rev + 1)) * 100
                fig4 = px.line(x=dates, y=cash_ratio_pct, markers=True, title="净现比 (%)")
                fig4.add_hline(y=100, line_dash="dash", line_color="green")
                fig4.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig4, use_container_width=True)

            max_spend = max(capex[latest], repay[latest], div[latest])
            spend_msg = ""
            if max_spend == capex[latest] and max_spend > 0:
                spend_msg = "🚀 **进取型** (扩产为主)"
            elif max_spend == repay[latest] and max_spend > 0:
                spend_msg = "🛡️ **防御型** (还债为主)"
            elif max_spend == div[latest] and max_spend > 0:
                spend_msg = "💰 **回报型** (分红为主)"
            st.info(f"💡 **AI 点评**：公司当前处于 {spend_msg} 阶段。")

            # === 模块四：结论 ===
            st.markdown("---")
            st.header("📝 最终审计结论")
            final_score = 60
            if cash_ratio_val > 1:
                final_score += 15
            elif cash_ratio_val < 0:
                final_score -= 10
            if core_profit[latest] / op_prof[latest] > 0.8:
                final_score += 15
            elif core_profit[latest] / op_prof[latest] < 0.5:
                final_score -= 10
            if div[latest] > 0: final_score += 10
            if total_loss[latest] < 0: final_score -= 5
            if op_ratio > 0.7:
                final_score += 5
            elif op_ratio < 0.5:
                final_score -= 5
            final_score = min(100, max(0, final_score))

            e1, e2 = st.columns([1, 3])
            color = "green" if final_score >= 80 else "orange" if final_score >= 60 else "red"
            e1.markdown(
                f"<div style='text-align:center; border:4px solid {color}; padding:20px; border-radius:10px'><h1 style='color:{color}; margin:0'>{final_score} 分</h1></div>",
                unsafe_allow_html=True)
            if final_score >= 80:
                e2.success("财务状况健康，主业清晰，分红积极，具备较高的长期投资价值。")
            elif final_score >= 60:
                e2.warning("财务状况尚可，但存在一些瑕疵，建议保持关注。")
            else:
                e2.error("财务风险较高，请谨慎决策！")
else:
    st.info("👈 请在左侧上传三个Excel报表开始体检")