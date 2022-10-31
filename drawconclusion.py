import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    sns.set(color_codes=True)

    cell = ['CinC_ECG_torso', 'Haptics', 'InlineSkate', 'MALLAT', 'Phoneme', 'UWaveGestureLibraryAll']
    pvalue1 = [3.07, 1.07, 1.06, 1.05, 1.02, 1.25]
    pvalue2 = [3.16,2.21, 2.01, 2.61, 1.76,3.01 ]
    temp = 0
    for i in range(len(pvalue1)):
        temp =temp +((pvalue2[i]-pvalue1[i])/pvalue1[i])
    print(temp/6)





    width = 0.20
    index = np.arange(len(cell))


    figsize = (10, 7)  # 调整绘制图片的比例
    #plt.plot(p1, p2, color='red', label='5% significance level')  # 绘制直线
    #plt.plot(q1, q2, color='yellow', label='10% significance level')  # 绘制直线
    # 若是不想显示直线，可以直接将上面两行注释掉
    plt.bar(index, pvalue1, width, color="salmon")  # 绘制柱状图
    plt.bar(index+0.2, pvalue2, width, color="#87CEFA")  # 绘制柱状图
    # plt.xlabel('cell type') #x轴
    plt.ylabel('Compression Ratio')  # y轴
    plt.title('Result of Comparison',x=0.5,y=1.1)  # 图像的名称
    plt.xticks(index+0.1, cell, fontsize=7)  # 将横坐标用cell替换,fontsize用来调整字体的大小
    labels = ['Rl-method', 'Ad-method']
    color = ['salmon','#87CEFA']
    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # 下面一行中bbox_to_anchor指定了legend的位置
    ax.legend(handles=patches, bbox_to_anchor=(0.95, 1.12), ncol=4)  # 生成legend


    plt.show()
