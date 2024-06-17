import matplotlib.pyplot as plt

# 示例数据
x = [ 3, 4,5,6,7]
y1 = [0.976, 0.992, 0.992, 0.984, 0.983]
x1 = [0.02,0.04,0.06,0.08,0.1]
y2 = [0.976,0.984,0.992,0.977,0.969]

# 创建图表和轴对象
fig, ax = plt.subplots()

# 绘制多条折线
# ax.plot(x, y1, marker='o',label='FiveDirections')

ax.plot(x1, y2, marker='o',color='#F3752C',label='FiveDirections')
# 添加图例
ax.legend()

# 添加标题和标签
ax.set_xticks(x1)
ax.set_yticks([0.972,0.976,0.980,0.984,0.988,0.992])
# ax.set_xticks(range(min(x), max(x) + 1, 1))
# ax.set_xlabel('number of layers')
ax.set_xlabel('rate of seeds')
ax.set_ylabel('ACC')

# 显示图表
plt.show()