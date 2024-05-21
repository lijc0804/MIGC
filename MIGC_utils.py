import numpy as np
import matplotlib.pyplot as plt
import scipy
import scanpy as sc
import scvelo as scv
import os
import anndata as ad
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
    
def batch_ERSMI(I1, I2, sigma=0.05, mu_min=0, mu_max=1, mu_step=0.1):
    import torch
    batch_size = I1.shape[0] 
    img_size = 1
    for tmp in range(1, len(I1.shape)):
        img_size = img_size * I1.shape[tmp] 
    
    def kernel_F(I, mu_list, sigma, n):
        tmp_mu = mu_list.view(1, -1, 1).repeat(batch_size, 1, img_size).cuda()
        tmp_I = I.view(batch_size, 1, -1).repeat(1, n*n, 1)
        tmp = tmp_mu - tmp_I
        mat = torch.exp(-tmp.pow(2) / (2 * sigma ** 2))
        return mat

    mu = torch.Tensor(torch.range(mu_min, mu_max, mu_step)).cuda()
    n = len(mu)
    x_mu_list = mu.repeat(n).view(-1, n*n)
    y_mu_list = mu.unsqueeze(0).t().repeat(1, n).view(-1, n*n)

    mat_K = kernel_F(I1, x_mu_list, sigma=sigma, n=n)
    mat_L = kernel_F(I2, y_mu_list, sigma=sigma, n=n)

    H1 = ((mat_K.matmul(mat_K.transpose(1,2))).mul(mat_L.matmul(mat_L.transpose(1,2))) / (img_size ** 2)).cuda()
    # h1 = (mat_K.mul(mat_L)).mm(torch.ones(img_size, 1)) / img_size

    H2 = ((mat_K.mul(mat_L)).matmul((mat_K.mul(mat_L)).transpose(1,2)) / img_size).cuda()
    h2 = ((mat_K.sum(2).view(batch_size,-1, 1)).mul(mat_L.sum(2).view(batch_size,-1, 1)) / (img_size ** 2)).cuda()
    # h2 = (((mat_K.sum(1).view(-1,1)).mul(mat_L.sum(1).view(-1,1)) / (img_size ** 2)).double()).cuda()

    H2 = 0.5 * H1 + 0.5 * H2
    tmp = H2 + 0.001 * torch.eye(H2.shape[1]).cuda()
    alpha = (tmp.inverse()).matmul(h2)
    ersmi = (2 * (alpha.transpose(1,2)).matmul(h2) - ((alpha.transpose(1,2)).matmul(H2)).matmul(alpha) - 1).squeeze()
    return ersmi


def show_xt(x, t, title, c=None, alpha=None, text=None, axis_off=False, save_path=None):
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111)
    if c is not None:
        ax1.scatter(t, x, color=c, alpha=alpha, label='Expression')
    else:
        ax1.scatter(t, x, c='blue', alpha=alpha, label='Expression')
    ax1.set_xlabel('t', fontdict={'size': 30})
    #ax1.set_ylabel('Expression', fontdict={'size': 20, 'color': 'black'})
    ax1.set_ylabel(title, fontdict={'size': 30, 'color': 'black'})

    plt.xticks([])
    plt.yticks([])
    plt.tick_params(axis='both', which='both', labelsize=20)
    #ax1.legend(loc='upper left')
    #mi_x = mutual_info_regression(x.reshape(-1, 1), t)[0]
    if not text == None:
        plt.text(0.05, 0.9, text, fontdict={'size': '20', 'color': 'Red'}, transform=plt.gca().transAxes)
    #plt.title(title, fontsize=25)
    if axis_off:
        ax1.axis('off')
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+'/'+title+'.png', dpi=300, bbox_inches='tight', pad_inches=0.0, format='png')
        plt.close()
    return 


def show_xc(x, clusters, title, x_label='Expression', y_label='Clusters', order=None, colors=None, text=None, figsize=None, save_path=None):
    df = pd.DataFrame({x_label: x, y_label: clusters})
    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    if order is not None:
        sns.boxplot(x=x_label, y=y_label, data=df, palette=colors, order=order, orient='h')
    else:
        sns.boxplot(x=x_label, y=y_label, data=df)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.tick_params(axis='both', which='both', labelsize=20)
    plt.title(title, fontsize=25)
    if text is not None:
        plt.text(0.8, 0.1, text, fontsize=15, transform=plt.gca().transAxes, ha='center', va='center', c='red')
    if save_path is not None:
        plt.savefig(save_path+'/'+title+'.png', dpi=300, bbox_inches='tight', pad_inches=0.0, format='png')
        plt.close()
    return


def GO_enrich(gene_list, background, organism='mouse', figure_path='figures/', save_name=''):
    import gseapy as gp
    #print(len(gene_list), len(background))
    if organism == 'mouse':
        gene_sets_KEGG='KEGG_2019_Mouse'
    elif organism == 'human':
        gene_sets_KEGG='KEGG_2019_Human'
    # GO
    result_go = gp.enrichr(gene_list=gene_list, 
                            organism=organism, 
                            gene_sets='GO_Biological_Process_2021', 
                            background=background,
                            # description='test', 
                            outdir=figure_path+'Enrichr_GO_BP_'+save_name)
    #result_go.res2d
    return


def KEGG_enrich(gene_list, background, organism='mouse', figure_path='figures/', save_name=''):
    import gseapy as gp
    #print(len(gene_list), len(background))
    if organism == 'mouse':
        gene_sets_KEGG='KEGG_2019_Mouse'
    elif organism == 'human':
        gene_sets_KEGG='KEGG_2019_Human'
    # KEGG
    result_kegg = gp.enrichr(gene_list=gene_list, 
                            organism=organism, 
                            gene_sets=gene_sets_KEGG, 
                            background=background,
                            #   description='test', 
                            outdir=figure_path+'Enrichr_KEGG_'+save_name)
    #result_kegg.res2d
    return


def while_GO_KEGG(gene_list, background, organism='mouse', figure_path='figures/', save_name=''):
    fail_counts = 0
    while 1:
        try:
            GO_enrich(gene_list, background, organism=organism, figure_path=figure_path, save_name=save_name)
            KEGG_enrich(gene_list, background, organism=organism, figure_path=figure_path, save_name=save_name)
            print('Finished GO KEGG')
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            fail_counts += 1
            print('Fail', fail_counts)
    return

def get_percentage(array, point):
    array = np.array(array)
    point = np.array(point)[0]
    sorted_indices = np.argsort(array)
    position_in_sorted = np.searchsorted(array[sorted_indices], point)
    rank = position_in_sorted + 1
    return round(rank/len(array)*100, 2)


def show_coplot(x, y, x_label='x', y_label='y'):
    #corr = np.corrcoef(x.reshape(1,-1), y.reshape(1,-1))[0,1]
    corr, p_value = spearmanr(x, y)
    _, ax = plt.subplots(figsize=(9, 9))
    data = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
    df = pd.DataFrame(data, columns=[x_label, y_label])
    #sns.kdeplot(data=df, x=x_label, y=y_label, fill=True, cmap='Blues', levels=5)
    sns.scatterplot(data=df, x=x_label, y=y_label, color='blue')
    plt.xticks(size=30)
    plt.yticks(size=30)
    #plt.axis('scaled')
    #ax.xaxis.set_major_locator(MultipleLocator(5))
    #ax.yaxis.set_major_locator(MultipleLocator(5))
    plt.text(x=0.4, y=0.05, s='Spearmanr: '+str(round(corr, 2)), size=35, transform=ax.transAxes)
    plt.xlabel(x_label, fontdict={'family':'Times New Roman', 'size':35})
    plt.ylabel(y_label, fontdict={'family':'Times New Roman', 'size':35})
    #plt.savefig(args.result_path + 'EM_Velocity_'+str(max_iter)+'.jpg', bbox_inches='tight', dpi=300) #pad_inches=0.0
    #plt.close()
    return

def show_X_t(X, t, genes, title, window_size, alpha=0.7):
    for i in range(X.shape[1]):
        x = X[:,i]
        x = x/x.max()
        data = pd.DataFrame({'Time': t, 'Data': x})
        #plt.plot(data['Time'], data['Data'], label=g)
        data['Smoothed_Data'] = data['Data'].rolling(window=window_size, min_periods=1).mean()
        plt.plot(data['Time'], data['Smoothed_Data'], label=genes[i], color='#4C72B0', alpha=alpha)#, marker='o')

    plt.xlabel('t', fontdict={'family':'Times New Roman', 'size':20})
    plt.ylabel(f'Normalized expression', fontdict={'family':'Times New Roman', 'size':20})
    #plt.legend()
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.title(title, fontsize=20)
    plt.show()
    plt.close()
    return