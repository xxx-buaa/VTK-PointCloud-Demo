#########
#
# 用于KDTree搜索和噪声点集可视化的程序
#
#########
import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from scipy.spatial import KDTree

class PointPickerInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, polydata, kdtree, colors, parent=None):
        # 调用父类构造函数
        super(PointPickerInteractorStyle, self).__init__()
        
        # 保存引用
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.polydata = polydata
        self.kdtree = kdtree
        self.colors = colors
        
        # 用于高亮显示被选择的点
        self.selected_point_actor = None

    def left_button_press_event(self, obj, event):
        # 获取鼠标点击位置
        click_pos = self.GetInteractor().GetEventPosition()

        # 创建拾取器
        picker = vtk.vtkPointPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.GetDefaultRenderer())

        # 获取被拾取点的 ID
        point_id = picker.GetPointId()

        if point_id != -1:
            # 获取点的坐标
            point = self.polydata.GetPoint(point_id)
            print(f"选中的点 ID: {point_id}, 坐标: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")

            # 高亮显示被选择的点
            self.highlight_selected_point(point)

        else:
            print("未选中任何点。")

        # 调用父类处理其他事件
        self.OnLeftButtonDown()
        return

    def highlight_selected_point(self, point):
        # 如果已有高亮显示的点，移除它
        if self.selected_point_actor:
            self.GetDefaultRenderer().RemoveActor(self.selected_point_actor)
            self.selected_point_actor = None

        # 创建一个小球来表示被选择的点
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(point)
        sphere_source.SetRadius(0.5)  # 根据点云尺度调整半径

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(1, 1, 1)  # 白色
        sphere_actor.GetProperty().SetOpacity(1.0)

        # 添加到渲染器
        self.GetDefaultRenderer().AddActor(sphere_actor)
        self.selected_point_actor = sphere_actor

        # 渲染更新
        self.GetInteractor().GetRenderWindow().Render()

def set_nearest_neighbors_red(kdtree, points, colors, query_point, k=1000):
    """
    将特定点的最近 k 个邻居的颜色设置为红色
    :param kdtree: scipy.spatial.KDTree
    :param points: numpy 数组，形状为 (n_points, 3)
    :param colors: vtkUnsignedCharArray
    :param query_point_index: 查询点的索引
    :param k: 最近邻的数量
    """
    
    # 查询最近 k+1 个点（包括自身）
    distances, indices = kdtree.query(query_point, k=k)

    # 设置邻居点的颜色为红色
    for idx in indices:
        colors.SetTuple3(idx, 255, 0, 0)
    
    # 标记颜色数组已修改
    colors.Modified()

    return indices

if __name__ == '__main__':

    # ===========================
    # 1. 读取 TXT 点云数据
    # ===========================

    # 读取 txt 文档
    source_data = np.loadtxt("src\CalibrationPointCloud.txt")

    """
    KDTree
    """
    kdtree = KDTree(source_data)

    # 新建 vtkPoints 实例
    points = vtk.vtkPoints()
    # 导入点数据
    points.SetData(numpy_to_vtk(source_data))

    # 新建 vtkPolyData 实例
    polydata = vtk.vtkPolyData()
    # 设置点坐标
    polydata.SetPoints(points)

    """
    颜色映射
    """
    # 创建一个基于高度的颜色映射
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    z_values = source_data[:, 2]
    z_min, z_max = z_values.min(), z_values.max()

    for z in z_values:
        normalized_z = (z - z_min) / (z_max - z_min) if z_max > z_min else 0
        # r = int(normalized_z * 255)
        # g = 0
        # b = int((1 - normalized_z) * 255)
        # colors.InsertNextTuple3(r, g, b)
        colors.InsertNextTuple3(122, 122, 122)

    # 将颜色数组添加到点数据
    polydata.GetPointData().SetScalars(colors)

    """
    通过点坐标查询最近的点集
    """
    query_point = [4.76, -71.17, 424.26]
    specific_point_index = 100
    neighbor_indices = set_nearest_neighbors_red(kdtree, source_data, colors, query_point, k=182)

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
    point_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
    point_actor.GetProperty().SetPointSize(0.5)     # 设置点大小  

    # ===========================
    # 3. 设置渲染器、渲染窗口和交互器
    # ===========================
    # 创建渲染器
    renderer = vtk.vtkRenderer()

    # 创建渲染窗口
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 1200)
    render_window.SetWindowName('点云可视化')

    # 创建渲染窗口交互器
    render_interactor = vtk.vtkRenderWindowInteractor()
    render_interactor.SetRenderWindow(render_window)
    #render.SetBackground(colors.GetColor3d('RosyBrown'))

    # 设置交互样式（可选）
    style = PointPickerInteractorStyle(polydata, kdtree, colors)
    style.SetDefaultRenderer(renderer)
    render_interactor.SetInteractorStyle(style)
    # render_interactor.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())

    # ===========================
    # 4. 添加演员到渲染器
    # ===========================
    renderer.AddActor(point_actor)  # 添加点云演员
    renderer.GradientBackgroundOn()
    renderer.SetBackground(0.7, 0.85, 1.0)   # 起始颜色
    renderer.SetBackground2(0.5, 0.75, 1.0)  # 结束颜色

    # ===========================
    # 5. 启动渲染循环
    # ===========================
    render_interactor.Initialize()
    render_window.Render()
    render_interactor.Start()
