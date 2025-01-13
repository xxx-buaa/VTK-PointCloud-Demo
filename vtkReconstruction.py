import numpy as np
from scipy.spatial import KDTree
from stl import mesh
import itertools

# 文件路径
input_file = 'src\\CalibrationPointCloud_registrated.txt'
output_file = 'src\\output.stl'

# 读取点云数据
def read_point_cloud(file_path):
    try:
        points = np.loadtxt(file_path)
        if points.shape[1] != 3:
            raise ValueError("点云数据应包含三个坐标 (x, y, z)。")
        return points
    except Exception as e:
        print(f"读取点云数据时出错: {e}")
        return None

# 贪婪三角剖分算法
def greedy_triangulation(points, k=10):
    tree = KDTree(points)
    triangles = set()
    for idx, point in enumerate(points):
        # 查询每个点的k个最近邻（包括自身）
        dist, neighbors = tree.query([point], k=k)
        neighbors = neighbors[0]
        # 生成所有可能的三角形
        for i in range(1, len(neighbors)):
            for j in range(i+1, len(neighbors)):
                a = neighbors[i]
                b = neighbors[j]
                # 确保点的索引升序，避免重复
                triangle = tuple(sorted([idx, a, b]))
                triangles.add(triangle)
    return np.array(list(triangles))

# 计算三角形法向量
def calculate_normals(points, triangles):
    normals = []
    for tri in triangles:
        p0, p1, p2 = points[tri]
        v1 = p1 - p0
        v2 = p2 - p0
        # 计算法向量
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            normal = np.array([0, 0, 0])
        else:
            normal = normal / norm
        normals.append(normal)
    return np.array(normals)

# 生成 STL 网格
def create_stl_mesh(points, triangles, normals):
    stl_mesh = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
    for i, tri in enumerate(triangles):
        for j in range(3):
            stl_mesh.vectors[i][j] = points[tri[j]]
        stl_mesh.normals[i] = normals[i]
    return stl_mesh

def main():
    points = read_point_cloud(input_file)
    if points is None:
        return

    print("点云数据读取成功。点的数量:", points.shape[0])

    print("构建 KD-Tree...")
    tree = KDTree(points)
    print("KD-Tree 构建完成。")

    print("开始贪婪三角剖分...")
    triangles = greedy_triangulation(points, k=10)
    print(f"生成三角形数量: {len(triangles)}")

    print("计算法向量...")
    normals = calculate_normals(points, triangles)

    print("创建 STL 网格...")
    stl_mesh = create_stl_mesh(points, triangles, normals)

    print(f"导出 STL 文件到 {output_file}...")
    stl_mesh.save(output_file)
    print("STL 文件导出完成。")

if __name__ == "__main__":
    main()
