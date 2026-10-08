# 导入Streamlit库，用于构建Web应用程序交互界面
import streamlit as st
# 导入NumPy库，用于执行高效的数组和矩阵运算
import numpy as np
# 导入Python内置的math库，用于执行阶乘计算（避免使用已废弃的np.math）
import math
# 导入SciPy的优化模块，用于寻找最佳的最接近球面半径（BFS）
from scipy.optimize import minimize
# 导入Pandas库，用于构建和展示表格数据
import pandas as pd
# 导入Plotly图形库，用于绘制高质量的三维交互式图表
import plotly.graph_objects as go

# 设置Streamlit页面的全局配置：网页标题、图标、以及宽屏布局
st.set_page_config(page_title="Zernike波面拟合小程序", layout="wide", page_icon="🔬")


# ================= 1. 基础数学函数定义 =================

# 定义计算标准Zernike多项式径向部分 R_n^l(rho) 的函数
def radial_poly(n, m, rho):
    # 计算m的值，对应公式中的m (m <= n)
    m_val = int((n - m) / 2)
    # 初始化径向多项式的结果为零数组
    R = np.zeros_like(rho)
    # 循环计算求和公式中的每一项
    for s in range(m_val + 1):
        # 计算分子：阶乘部分 (n-s)!
        num = math.factorial(n - s)
        # 计算分母：阶乘部分 s! * (m-s)! * (n-m-s)!
        den = math.factorial(s) * math.factorial(m_val - s) * math.factorial(n - m_val - s)
        # 根据公式累加每一项：(-1)^s * 系数 * rho的幂次
        R += ((-1) ** s) * (num / den) * (rho ** (n - 2 * s))
    # 返回计算完成的径向多项式结果
    return R

# 定义计算完整Zernike多项式（包含角度部分）的函数
def zernike_poly(n, m, rho, theta):
    # 如果方位角频率 m 大于 0
    if m > 0:
        # 返回包含余弦项的多项式
        return radial_poly(n, m, rho) * np.cos(m * theta)
    # 如果方位角频率 m 小于 0
    elif m < 0:
        # 返回包含正弦项的多项式
        return radial_poly(n, -m, rho) * np.sin(-m * theta)
    # 如果方位角频率 m 等于 0
    else:
        # 只返回径向部分的多项式
        return radial_poly(n, 0, rho)

# 定义生成前N项Zernike多项式索引（使用Fringe顺序）的函数
def get_zernike_basis_terms(num_terms=15):
    # 定义包含(n, m)元组的列表，对应Fringe Zernike的排列顺序
    indices = [
        (0, 0),   # Z1: Piston (平移)
        (1, 1),   # Z2: X tilt (X倾斜)
        (1, -1),  # Z3: Y tilt (Y倾斜)
        (2, 0),   # Z4: Defocus (离焦)
        (2, -2),  # Z5: Astigmatism 45 (45度像散)
        (2, 2),   # Z6: Astigmatism 0 (0度像散)
        (3, -1),  # Z7: Y coma (Y彗差)
        (3, 1),   # Z8: X coma (X彗差)
        (3, -3),  # Z9: Y trefoil (Y三叶草)
        (3, 3),   # Z10: X trefoil (X三叶草)
        (4, 0),   # Z11: Primary Spherical (初级球差)
        (4, 2),   # Z12: Secondary Astigmatism (二级像散)
        (4, -2),  # Z13: Secondary Astigmatism (二级像散)
        (4, 4),   # Z14: Quadrafoil (四叶草)
        (4, -4)   # Z15: Quadrafoil (四叶草)
    ]
    # 截取并返回前 num_terms 项索引
    return indices[:num_terms]

# 定义计算非球面矢高（Sag）的函数
def calculate_asphere_sag(x, y, r, k, coes):
    # 计算径向距离的平方 S^2 = x^2 + y^2
    S2 = x**2 + y**2
    # 定义一个极小的数，防止分母为0
    eps = 1e-10
    # 计算分母部分：1 + sqrt(1 - (k+1)S^2/r^2)，使用maximum防止根号下出现负数
    denom = 1 + np.sqrt(np.maximum(0, 1 - (k + 1) * S2 / (r**2)))
    # 计算二次曲面（Conic）部分的矢高
    sag_conic = (S2 / r) / (denom + eps)
    # 初始化高阶非球面项矢高为零
    sag_high = np.zeros_like(S2)
    # 遍历高阶非球面系数 (a4, a6, a8...)
    for i, coe in enumerate(coes):
        # 累加高阶项，i=0对应(x^2+y^2)^2
        sag_high += coe * (S2 ** (i + 2))
    # 返回总非球面矢高
    return sag_conic + sag_high

# 定义计算最接近球面（BFS）半径的函数
def calculate_bfs(x, y, sag_asphere):
    # 定义优化算法的目标函数：计算给定半径R_bfs的球面与原始非球面的均方误差
    def objective(R_bfs):
        # 计算径向距离的平方
        S2 = x**2 + y**2
        # 计算球面矢高：Z = R - sqrt(R^2 - S^2)，使用maximum防止负数
        sag_sphere = R_bfs - np.sqrt(np.maximum(0, R_bfs**2 - S2))
        # 返回矢高差值的均方误差
        return np.mean((sag_sphere - sag_asphere)**2)
    # 使用SciPy的minimize函数，寻找使目标函数最小化的R_bfs
    res = minimize(objective, x0=2000.0, bounds=[(1.0, 10000.0)])
    # 返回优化后的最接近球面半径
    return res.x[0]

# 定义Gram-Schmidt正交化函数
def gram_schmidt_orthogonalize(Z_matrix, weights):
    # 获取Zernike多项式的项数
    num_terms = Z_matrix.shape[1]
    # 初始化正交多项式矩阵F
    F_matrix = np.zeros_like(Z_matrix)
    # 遍历每一项Zernike多项式
    for j in range(num_terms):
        # 初始的G向量为当前的Zernike多项式
        G_j = Z_matrix[:, j].copy()
        # 减去它在前j-1个已正交化多项式上的投影
        for k in range(j):
            # 计算内积 <Z_j, F_k>
            inner_prod = np.sum(weights * Z_matrix[:, j] * F_matrix[:, k])
            # 计算投影系数（因为F_k已归一化，分母为1）
            G_jk = -inner_prod
            # 从当前的G_j中减去投影
            G_j += G_jk * F_matrix[:, k]
        # 计算G_j的L2范数
        norm_Fj = np.sqrt(np.sum(weights * G_j**2))
        # 如果范数极小，说明线性相关，直接置零
        if norm_Fj < 1e-12:
            F_matrix[:, j] = 0
        # 否则进行归一化
        else:
            F_matrix[:, j] = G_j / norm_Fj
    # 返回正交化后的多项式矩阵
    return F_matrix


# ================= 2. 光瞳与网格生成 =================

# 定义生成光瞳掩膜的函数
def generate_pupil_mask(X, Y, pupil_type, e, x_offset_norm, y_offset_norm, a=1/np.sqrt(1.5)):
    # 将网格点减去归一化的偏移量，得到光瞳的局部X坐标
    x_local = X - x_offset_norm
    # 将网格点减去归一化的偏移量，得到光瞳的局部Y坐标
    y_local = Y - y_offset_norm
    # 计算局部极坐标的径向距离平方
    r2 = x_local**2 + y_local**2
    
    # 如果选择圆形光瞳
    if pupil_type == "Circular":
        # 返回径向距离小于等于1的掩膜
        return r2 <= 1.0
    # 如果选择环形光瞳
    elif pupil_type == "Annular":
        # 返回径向距离在 e 和 1 之间的掩膜
        return (r2 >= e**2) & (r2 <= 1.0)
    # 如果选择正方形光瞳
    elif pupil_type == "Square":
        # 返回内接正方形的掩膜
        return (np.abs(x_local) <= 1/np.sqrt(2)) & (np.abs(y_local) <= 1/np.sqrt(2))
    # 如果选择矩形光瞳
    elif pupil_type == "Rectangle":
        # 返回矩形光瞳的掩膜
        return (np.abs(x_local) <= a) & (np.abs(y_local) <= np.sqrt(1 - a**2))
    # 如果选择正六边形光瞳
    elif pupil_type == "Hexagon":
        # 返回正六边形光瞳的掩膜
        return (np.abs(x_local) <= 1 - np.abs(y_local)/np.sqrt(3)) & (np.abs(y_local) <= np.sqrt(3)/2)
    # 如果类型未知，返回全False的掩膜
    return np.zeros_like(X, dtype=bool)

# 使用Streamlit缓存装饰器，避免重复计算网格数据
@st.cache_data
def generate_data(num_points, pupil_type, e, x_offset_norm, y_offset_norm):
    # 计算一维网格的分辨率（点数）
    n_grid = int(np.sqrt(num_points))
    # 如果网格分辨率小于2，直接返回空值，防止后续数组为空
    if n_grid < 2:
        return None, None, None, None, None, None, None, None
        
    # 生成X方向的一维坐标，范围固定在[-1.2, 1.2]以留出边界
    x_1d = np.linspace(-1.2, 1.2, n_grid)
    # 生成Y方向的一维坐标，范围固定在[-1.2, 1.2]以留出边界
    y_1d = np.linspace(-1.2, 1.2, n_grid)
    # 生成二维网格坐标矩阵
    X, Y = np.meshgrid(x_1d, y_1d)
    
    # 根据参数生成光瞳掩膜（包含偏移量处理）
    mask = generate_pupil_mask(X, Y, pupil_type, e, x_offset_norm, y_offset_norm)
    
    # 提取光瞳内的X坐标（展平为一维数组）
    x_p = X[mask]
    # 提取光瞳内的Y坐标（展平为一维数组）
    y_p = Y[mask]
    
    # 如果光瞳内没有点（例如偏移量太大），直接返回空值
    if len(x_p) == 0:
        return X, Y, mask, None, None, None, None, None
        
    # 计算光瞳局部坐标下的X坐标
    x_local = x_p - x_offset_norm
    # 计算光瞳局部坐标下的Y坐标
    y_local = y_p - y_offset_norm
    
    # 计算径向极坐标 rho
    rho = np.sqrt(x_local**2 + y_local**2)
    # 计算角度极坐标 theta
    theta = np.arctan2(y_local, x_local)
    
    # 计算网格步长 dx
    dx = 2.4 / (n_grid - 1) if n_grid > 1 else 1.0
    # 计算光瞳面积 area
    area = np.sum(mask) * dx**2
    # 计算积分权重（用于正交化）
    weights = (1.0 / area) * np.ones_like(x_p)
    
    # 返回生成的网格、掩膜及一维坐标数据
    return X, Y, mask, x_p, y_p, rho, theta, weights


# ================= 3. 主程序入口 =================

# 定义主函数
def main():
    # 在页面顶部显示主标题
    st.title("Zernike多项式波面拟合小程序")
    # 在页面顶部显示副标题/说明
    st.markdown("仿照Mathematica界面，支持多种光瞳类型和正交化算法。基于Streamlit部署。")

    # 使用侧边栏创建参数输入区
    with st.sidebar:
        # 侧边栏标题
        st.header("参数设置")
        # 输入镜面直径
        diameter = st.number_input("Diameter (直径)", value=500.0, step=10.0)
        # 输入X方向偏移量（默认设为0以避免初始为空）
        x_offset = st.number_input("X Offset (X偏移)", value=0.0, step=1.0)
        # 输入Y方向偏移量（默认设为0以避免初始为空）
        y_offset = st.number_input("Y Offset (Y偏移)", value=0.0, step=1.0)
        # 输入顶点曲率半径
        roc = st.number_input("Radius of Curvature (顶点曲率半径)", value=2000.0, step=100.0)
        # 输入二次曲面常数
        conic = st.number_input("Conic Constant (二次曲面常数)", value=-1.0, step=0.1)
        # 输入计算步长（影响采样点数量）
        step = st.number_input("Computing Step (计算步长)", value=10, step=1)
        
        # 添加分隔线
        st.markdown("---")
        # 提示高阶非球面系数输入
        st.write("高阶非球面系数")
        # 输入高阶非球面系数a4
        a4 = st.number_input("a4", value=0.0, format="%.2e")
        # 输入高阶非球面系数a6
        a6 = st.number_input("a6", value=0.0, format="%.2e")
        # 输入高阶非球面系数a8
        a8 = st.number_input("a8", value=0.0, format="%.2e")
        # 输入高阶非球面系数a10
        a10 = st.number_input("a10", value=0.0, format="%.2e")
        # 输入高阶非球面系数a12
        a12 = st.number_input("a12", value=0.0, format="%.2e")
        
        # 添加分隔线
        st.markdown("---")
        
        # 定义光瞳类型的中文映射字典（显示中文，底层传英文值）
        pupil_options = {
            "Circular": "Circular（圆形）",
            "Annular": "Annular（环形）",
            "Square": "Square（方形）",
            "Rectangle": "Rectangle（矩形）",
            "Hexagon": "Hexagon（六边形）"
        }
        # 使用format_func将选项显示为中文，实际传给变量的值依然是英文
        pupil_type = st.selectbox(
            "pupilname (光瞳类型)", 
            options=list(pupil_options.keys()), 
            format_func=lambda x: pupil_options[x]
        )
        
        # 输入环形光瞳的遮拦比
        e = st.slider("遮拦比 e (仅环形光瞳)", 0.0, 0.9, 0.3, 0.05)
        # 复选框：是否使用真实正交Zernike多项式
        use_real_zernike = st.checkbox("useRealZernike", value=True)
        # 复选框：是否显示残差图
        show_residual = st.checkbox("showResidual", value=True)
        # 滑块：选择拟合的Zernike项数
        num_terms = st.slider("拟合项数 (Zernike Terms)", 3, 15, 9)

    # 检查计算步长是否合法（必须大于0）
    if step <= 0:
        # 如果不合法，显示错误信息并终止运行
        st.error("Computing Step 必须大于 0！")
        # 返回，终止后续计算
        return
    # 计算总采样点数
    num_points = int((diameter / step) ** 2)
    # 检查采样点数是否过小
    if num_points < 4:
        # 如果过小，显示错误信息并终止运行
        st.error("网格分辨率太低。请减小 Computing Step 或增大 Diameter。")
        # 返回，终止后续计算
        return
        
    # 将X偏移量转换为归一化偏移量（除以半径）
    x_offset_norm = x_offset / (diameter / 2)
    # 将Y偏移量转换为归一化偏移量（除以半径）
    y_offset_norm = y_offset / (diameter / 2)

    # 调用generate_data函数生成网格和数据
    X, Y, mask, x_p, y_p, rho, theta, weights = generate_data(num_points, pupil_type, e, x_offset_norm, y_offset_norm)
    
    # 检查光瞳内是否有采样点
    if x_p is None or len(x_p) == 0:
        # 如果为空，显示友好的错误提示
        st.error("当前参数下光瞳掩膜为空！可能是 X/Y Offset 太大，导致光瞳移出了计算区域。请尝试将 Offset 设为 0，或减小 Computing Step。")
        # 返回，终止后续计算
        return

    # 将高阶非球面系数组合成列表
    coes = [a4, a6, a8, a10, a12]
    # 计算非球面面型的矢高
    sag_asphere = calculate_asphere_sag(x_p, y_p, roc, conic, coes)
    
    # 计算最接近球面半径（BFS）
    bfs_radius = calculate_bfs(x_p, y_p, sag_asphere)
    # 计算径向距离平方
    S2 = x_p**2 + y_p**2
    # 计算最接近球面的矢高
    sag_bfs = bfs_radius - np.sqrt(np.maximum(0, bfs_radius**2 - S2))
    
    # 计算非球面与最接近球面的偏差
    delta_z = sag_asphere - sag_bfs
    
    # 获取Zernike多项式的索引
    zernike_indices = get_zernike_basis_terms(num_terms)
    # 初始化Zernike多项式值矩阵
    Z_matrix = np.zeros((len(x_p), num_terms))
    # 遍历所有Zernike项
    for i, (n, m) in enumerate(zernike_indices):
        # 计算每一项Zernike多项式在采样点上的值
        Z_matrix[:, i] = zernike_poly(n, m, rho, theta)
        
    # 如果选择了使用正交化
    if use_real_zernike:
        # 调用Gram-Schmidt正交化函数
        U_matrix = gram_schmidt_orthogonalize(Z_matrix, weights)
    # 如果未选择正交化
    else:
        # 直接使用标准的Zernike多项式矩阵
        U_matrix = Z_matrix
        
    # 使用伪逆矩阵求解最小二乘问题
    U_pinv = np.linalg.pinv(U_matrix)
    # 计算Zernike系数
    z_coeffs = np.dot(U_pinv, delta_z)
    # 将Zernike系数转换为波长单位（纳米）
    z_coeffs_nm = z_coeffs * 1000 / 0.6328
    
    # 计算拟合面型矢高
    fitted_sag = np.dot(U_matrix, z_coeffs)
    # 计算拟合残差
    residual = delta_z - fitted_sag
    
    # 计算拟合面型的PV值
    pv_fitted = np.max(fitted_sag) - np.min(fitted_sag)
    # 计算拟合面型的RMS值
    rms_fitted = np.sqrt(np.mean(fitted_sag**2))
    # 计算残差的PV值
    pv_residual = np.max(residual) - np.min(residual)
    # 计算残差的RMS值
    rms_residual = np.sqrt(np.mean(residual**2))
    # 计算去除平移项后的Zernike系数平方和开根号
    sqrt_z2_sum = np.sqrt(np.sum(z_coeffs_nm[1:]**2))
    # 获取平移项（Piston）系数
    mean_z = z_coeffs_nm[0]

    # 创建两列布局，比例1:2
    col1, col2 = st.columns([1, 2])
    
    # 左侧列显示数据表格和统计指标
    with col1:
        # 显示子标题
        st.subheader("拟合结果数据表")
        # 定义像差名称列表
        terms_names = ["Piston", "X tilt", "Y tilt", "Defocus", "45°Astigmatism", 
                       "90°Astigmatism", "X Coma", "Y Coma", "3rd Spherical",
                       "Sec Astig 45", "Sec Astig 0", "Quadrafoil 45", "Quadrafoil 0",
                       "Sec Trefoil 45", "Sec Trefoil 0"]
        # 构建Pandas数据框
        df_results = pd.DataFrame({
            # 截取对应项数的名称
            "Aberration names": terms_names[:num_terms],
            # Zernike系数值，保留4位小数
            "Zernike Coefficients (nm)": z_coeffs_nm.round(4)
        })
        # 在界面上显示数据框
        st.dataframe(df_results, use_container_width=True)
        
        # 显示子标题
        st.subheader("统计指标")
        # 显示BFS半径
        st.write(f"最接近球面半径 (BFS Radius): {bfs_radius:.2f} mm")
        # 显示拟合面型PV值
        st.write(f"拟合面型 PV: {pv_fitted:.4f} mm")
        # 显示拟合面型RMS值
        st.write(f"拟合面型 RMS: {rms_fitted:.4f} mm")
        # 显示残差PV值
        st.write(f"残差 PV: {pv_residual:.4f} mm")
        # 显示残差RMS值
        st.write(f"残差 RMS: {rms_residual:.4f} mm")
        # 显示平均Piston值
        st.write(f"Mean (Piston): {mean_z:.4f} nm")
        # 显示去除Piston后的RMS
        st.write(f"Sqrt(Z2^2 - Zn^2): {sqrt_z2_sum:.4f} nm")

    # 右侧列显示三维图形
    with col2:
        # 显示子标题
        st.subheader("3D 面型与偏差图")
        
        # 初始化二维数组用于绘图，用NaN填充非光瞳区域
        Z_asphere_2d = np.full(X.shape, np.nan)
        # 将一维非球面矢高数据填入二维数组的对应位置
        Z_asphere_2d[mask] = sag_asphere
        
        # 初始化二维数组用于绘图，用NaN填充非光瞳区域
        Z_delta_2d = np.full(X.shape, np.nan)
        # 将一维偏差数据填入二维数组的对应位置
        Z_delta_2d[mask] = delta_z
        
        # 初始化二维数组用于绘图，用NaN填充非光瞳区域
        Z_residual_2d = np.full(X.shape, np.nan)
        # 将一维残差数据填入二维数组的对应位置
        Z_residual_2d[mask] = residual
        
        # 创建原始非球面的3D曲面图
        fig1 = go.Figure(data=[go.Surface(z=Z_asphere_2d, x=X, y=Y, colorscale='Viridis')])
        # 更新图表标题和坐标轴标签
        fig1.update_layout(title="Original Surface", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        # 在Streamlit中显示图表
        st.plotly_chart(fig1, use_container_width=True)
        
        # 创建偏差图的3D曲面图
        fig2 = go.Figure(data=[go.Surface(z=Z_delta_2d, x=X, y=Y, colorscale='Jet')])
        # 更新图表标题和坐标轴标签
        fig2.update_layout(title="3D Map of Departure", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Delta Z'))
        # 在Streamlit中显示图表
        st.plotly_chart(fig2, use_container_width=True)
        
        # 如果勾选了显示残差
        if show_residual:
            # 显示子标题
            st.subheader("拟合残差分布")
            # 创建残差二维等高线图
            fig3 = go.Figure(data=[go.Contour(z=Z_residual_2d, x=X[0,:], y=Y[:,0], colorscale='Jet', contours=dict(showlabels=True))])
            # 更新图表标题，包含PV和RMS值
            fig3.update_layout(title=f"Fitting Residual: PV={pv_residual:.4f}, RMS={rms_residual:.4f}")
            # 在Streamlit中显示图表
            st.plotly_chart(fig3, use_container_width=True)

# 判断当前脚本是否作为主程序运行
if __name__ == "__main__":
    # 调用主函数
    main()