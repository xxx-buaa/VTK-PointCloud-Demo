import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

# ==============================
# 辅助函数: 旋转和位移变换
# ==============================
def apply_transformation(points, rotation_matrix, translation_vector):
    """
    对点云进行旋转和平移变换
    :param points: 输入点云数据 (Nx3)
    :param rotation_matrix: 旋转矩阵 (3x3)
    :param translation_vector: 平移向量 (3,)
    :return: 变换后的点云数据
    """
    # transformed_points = np.dot(points, rotation_matrix.T) + translation_vector
    transformed_points = np.dot(points, rotation_matrix.T) + translation_vector
    return transformed_points

if __name__ == '__main__':

    # ===========================
    # 1. 读取 TXT 点云数据
    # ===========================

    # 读取 txt 文档
    # source_data = np.loadtxt("src\CalibrationPointCloud_denoised.txt")
    source_data = np.loadtxt("src\\point_cloud.txt")
    # source_data = np.loadtxt("平面度\\探测点组-1.txt", delimiter=',')
    print(f"初始点云数量：{len(source_data)}")

    # ===========================
    # 2. 点云变换 (旋转与平移)
    # ===========================
    # 定义旋转矩阵（绕Z轴旋转 45°）
    theta = np.radians(145)  # 旋转角度（弧度）
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [-sin_theta,  -cos_theta, 0],
        [0, 0, -1]
    ])

    # 定义平移向量
    translation_vector = np.array([10, 0, 750])  # 沿X轴平移5个单位
    transformed_data = (source_data @ rotation_matrix.T) + translation_vector
    # 应用变换
    # transformed_data = apply_transformation(source_data, rotation_matrix, translation_vector)

    # 新建 vtkPoints 实例
    points = vtk.vtkPoints()
    # 导入点数据
    # points.SetData(numpy_to_vtk(source_data))
    points.SetData(numpy_to_vtk(transformed_data))
    # 新建 vtkPolyData 实例
    polydata = vtk.vtkPolyData()
    # 设置点坐标
    polydata.SetPoints(points)

    # 顶点相关的 filter
    vertex_filter = vtk.vtkVertexGlyphFilter()
    vertex_filter.SetInputData(polydata)
    vertex_filter.Update()

    # mapper 实例
    point_mapper = vtk.vtkPolyDataMapper()
    # 关联 filter 输出
    point_mapper.SetInputConnection(vertex_filter.GetOutputPort())

    # actor 实例
    point_actor = vtk.vtkActor()
    # 关联 mapper
    point_actor.SetMapper(point_mapper)
    # 红色点显示
    # point_actor.GetProperty().SetColor(0.902, 0.624, 0.0) # 黄色
    point_actor.GetProperty().SetColor(0.0, 0.447, 0.698)
    point_actor.GetProperty().SetPointSize(0.5)     # 设置点大小

    # ===========================
    # 2. 读取 STL 文件
    # ===========================
    # 创建 STL 读取器
    stl_reader = vtk.vtkSTLReader()
    # stl_reader.SetFileName("src\YH0700000__1756220035.stl")  # STL 文件路径
    stl_reader.SetFileName("src\\output.stl")
    stl_reader.Update()

    # 创建 STL 的 mapper
    stl_mapper = vtk.vtkPolyDataMapper()
    stl_mapper.SetInputConnection(stl_reader.GetOutputPort())

    # 创建 STL 的 actor
    stl_actor = vtk.vtkActor()
    stl_actor.SetMapper(stl_mapper)
    # stl_actor.GetProperty().SetColor(0.902, 0.624, 0.0)  # 红色
    stl_actor.GetProperty().SetColor(0.0, 0.447, 0.698)  # 红色
    stl_actor.GetProperty().SetOpacity(1)    # 可选：设置透明度    

    # ===========================
    # 3. 设置渲染器、渲染窗口和交互器
    # ===========================
    # 创建渲染器
    renderer = vtk.vtkRenderer()
    # renderer.GradientBackgroundOn()

    # 创建渲染窗口
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 1200)
    render_window.SetWindowName('点云与 STL 可视化')

    # 创建渲染窗口交互器
    render_interactor = vtk.vtkRenderWindowInteractor()
    render_interactor.SetRenderWindow(render_window)
    #render.SetBackground(colors.GetColor3d('RosyBrown'))

    # 设置交互样式（可选）
    render_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    # ===========================
    # 4. 添加演员到渲染器
    # ===========================
    renderer.AddActor(point_actor)  # 添加点云演员
    # renderer.AddActor(stl_actor)    # 添加 STL 演员
    # renderer.SetBackground(0.7, 0.85, 1.0)   # 起始颜色
    # renderer.SetBackground2(0.5, 0.75, 1.0)  # 结束颜色
    renderer.SetBackground(1, 1, 1)

    # ===========================
    # 5. 启动渲染循环
    # ===========================
    render_interactor.Initialize()
    render_window.Render()
    render_interactor.Start()
