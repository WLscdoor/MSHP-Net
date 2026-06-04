import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path='confusion_matrix.png', dpi=300):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

    print(f"Consfusion matrix saved")
    

def draw_tsne(x_sne, all_labels, labels_name, fontsize = 16, perplexity = 50):
        x_sne = np.array(x_sne)
        label_sne = np.array(all_labels)
        index_to_name = { idx: name for name, idx in labels_name.items() }
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(np.array(x_sne))
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                            c=label_sne, cmap='tab10', alpha=0.7)
        unique_labels = np.unique(label_sne)
        handles = []
        legend_labels = []
        for lbl in unique_labels:
            color = scatter.cmap(scatter.norm(lbl))
            handles.append(
                Line2D([0], [0],
                    marker='o',
                    color='w',
                    markerfacecolor=color,
                    markersize=8)
            )
            legend_labels.append(index_to_name[lbl])
        plt.legend(handles, legend_labels, title="Classes")
        plt.xlabel('Dimension 1', fontsize = fontsize)
        plt.ylabel('Dimension 2', fontsize = fontsize)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./T_SNE', dpi=600)
        plt.close()
        print(f'T-SNE saved')



