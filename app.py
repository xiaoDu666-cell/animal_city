from flask import Flask, render_template,request,jsonify
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import io 
import base64

app=Flask(__name__)
# 定义角色五行向量数据库
character_vectors = {
    '朱迪 Judy (木行领袖)': [90, 70, 60, 50, 30],  # [木，火，土，金，水]
    '尼克 Nick (火行智囊)': [40, 90, 50, 30, 70],
    '本杰明 Benjamin (土行守护者)': [30, 60, 90, 70, 40],
    '牛局长 Chief Bogo (金行执法者)': [20, 30, 70, 90, 60],
    '新角色：阴谋家? (水行谋士)': [30, 40, 50, 60, 90],  # 为《疯狂动物城2》设计的悬念角色
}
df_characters = pd.DataFrame(character_vectors).T
df_characters.columns = ['木_行动力', '火_热情度', '土_稳定度', '金_规则力', '水_沉潜度']
print("角色五行向量数据库：")
print(df_characters)

# 2. 修正后的测试题定义（options必须是字典，包含选项文本和对应的分数向量）
questions = [
    {
        "question": "1. 面对一个全新的挑战项目，你的第一反应是？",
        "options": {
            "A. 立即制定计划，马上行动！": [10, 0, 0, 5, 0],    # 木+， 金+
            "B. 拉上朋友一起，边玩边做才有意思！": [5, 10, 0, 0, 0], # 木+， 火+
            "C. 先评估风险，确保每一步都稳妥。": [0, 0, 10, 5, 0],  # 土+， 金+
            "D. 独自深思，想清楚底层逻辑再动手。": [0, 0, 0, 0, 10]  # 水+
        }
    },
    {
        "question": "2. 在团队中，你通常扮演什么角色？",
        "options": {
            "A. 冲锋在前的开拓者。": [10, 5, 0, 0, 0],
            "B. 点燃气氛的凝聚者。": [0, 10, 5, 0, 0],
            "C. 稳定后方的支持者。": [0, 0, 10, 5, 0],
            "D. 制定规则的协调者。": [0, 0, 5, 10, 0]
        }
    },
    {
        "question": "3. 周末你更愿意如何度过？",
        "options": {
            "A. 尝试新的运动或户外探险。": [10, 5, 0, 0, 0],
            "B. 和朋友聚会，参加社交活动。": [0, 10, 5, 0, 0],
            "C. 在家整理房间，享受规律生活。": [0, 0, 10, 0, 5],
            "D. 研究一个感兴趣的理论或技术。": [0, 0, 0, 5, 10]
        }
    },
    {
        "question": "4. 遇到难题时，你倾向于？",
        "options": {
            "A. 快速试错，在行动中调整。": [10, 0, 0, 0, 0],
            "B. 求助他人，集思广益。": [5, 10, 0, 0, 0],
            "C. 按部就班，用已有方法解决。": [0, 0, 10, 5, 0],
            "D. 深入分析，找到根本原因。": [0, 0, 0, 5, 10]
        }
    },
    {
        "question": "5. 你如何做重要决定？",
        "options": {
            "A. 凭直觉快速决定。": [10, 5, 0, 0, 0],
            "B. 和信任的人讨论后决定。": [0, 10, 5, 0, 0],
            "C. 列出优缺点，谨慎选择。": [0, 0, 5, 10, 0],
            "D. 收集大量信息后深思熟虑。": [0, 0, 0, 5, 10]
        }
    },
    {
        "question": "6. 在《疯狂动物城》中，你最认同？",
        "options": {
            "A. 朱迪的勇敢追梦。": [10, 5, 0, 0, 0],
            "B. 尼克的灵活机智。": [0, 10, 5, 0, 0],
            "C. 本杰明的忠诚可靠。": [0, 0, 10, 0, 5],
            "D. 牛局长的坚守原则。": [0, 0, 0, 10, 5]
        }
    }
]

# 3. 修正后的计算函数
def calculate_user_vector(choices):
    user_vector = np.array([0, 0, 0, 0, 0])  # 初始向量
    
    for i, choice in enumerate(choices):
        # 获取选项列表
        option_keys = list(questions[i]["options"].keys())
        
        # 将A,B,C,D转换为索引 (0,1,2,3)
        if choice.upper() in ['A', 'B', 'C', 'D']:
            idx = ord(choice.upper()) - 65  # A->0, B->1, C->2, D->3
        else:
            # 如果输入的不是A-D，默认选第一个
            idx = 0
        
        # 确保索引不越界
        if idx < len(option_keys):
            option_key = option_keys[idx]
            # 累加分数向量
            user_vector += np.array(questions[i]["options"][option_key])
        else:
            print(f"警告：第{i+1}题选项索引{idx}超出范围")
    
    # 将得分归一化到0-100的区间
    if user_vector.max() > 0:
        user_vector = (user_vector / user_vector.max()) * 100
    
    return user_vector.round()

# 4. 测试用的用户选择（6个答案，对应A,B,C,D）
user_choices = ['A', 'B', 'C', 'D', 'A', 'B']  # 你可以修改这里的答案进行测试
user_vector = calculate_user_vector(user_choices)

print("✅ 计算成功！")
print(f"\n你的五行性格向量是：{user_vector}")
print(f"对应维度：[木-行动力, 火-热情度, 土-稳定度, 金-规则力, 水-沉潜度]")

def find_best_match(user_vec, char_df):
    best_char = None
    best_score = -1
    best_similarity_type = ""
    
    # 方法1: 余弦相似度 (值越大越相似，范围[-1,1])
    from numpy.linalg import norm
    similarities_cos = {}
    for char, vec in char_df.iterrows():
        cos_sim = np.dot(user_vec, vec) / (norm(user_vec) * norm(vec) + 1e-8) # 防止除以0
        similarities_cos[char] = cos_sim
        
    # 方法2: 欧氏距离的倒数 (距离越小越相似，取倒数让值越大越好)
    similarities_inv_dist = {}
    for char, vec in char_df.iterrows():
        distance = norm(np.array(user_vec) - np.array(vec))
        similarities_inv_dist[char] = 1 / (distance + 1) # 加1防止除零
    
    # 选择匹配度最高的角色 (这里以余弦相似度为例)
    best_char = max(similarities_cos, key=similarities_cos.get)
    best_score = similarities_cos[best_char]
    best_similarity_type = "余弦相似度"
    
    print(f"\n【匹配结果】")
    print(f"你的最佳匹配角色是：{best_char}")
    print(f"匹配度({best_similarity_type})：{best_score:.2%}")
    print(f"\n详细匹配度对比：")
    for char, score in similarities_cos.items():
        print(f"  {char}: {score:.2%}")
    
    return best_char, best_score, similarities_cos

best_match, match_score, all_scores = find_best_match(user_vector, df_characters)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager

# ====== 1. 添加中文字体支持 ======
# 方法1：使用系统自带的中文字体（推荐）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']  # 多个字体备选
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 方法2：如果上面的不行，指定具体字体路径（更稳定）
# font_path = r'C:\Windows\Fonts\msyh.ttc'  # 微软雅黑路径
# font_prop = font_manager.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()

# ====== 2. 修正雷达图函数 ======
def plot_radar_comparison(user_vec, char_vec, char_name):
    """
    生成用户和角色的雷达图对比
    
    参数:
    user_vec: 用户的五行向量 [木, 火, 土, 金, 水]
    char_vec: 角色的五行向量 [木, 火, 土, 金, 水]
    char_name: 角色名称
    """
    labels = ['行动力(木)', '热情度(火)', '稳定度(土)', '规则力(金)', '沉潜度(水)']
    num_vars = len(labels)
    
    # 计算雷达图的角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # 转换为列表并闭合图形
    user_vec = list(user_vec)
    char_vec = list(char_vec)
    user_vec += user_vec[:1]  # 闭合图形
    char_vec += char_vec[:1]  # 修正：原来是 [vec[:1]]
    angles += angles[:1]      # 修正：原来是 ang[1:]
    labels_closed = labels + [labels[0]]  # 闭合标签
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # 绘制用户数据
    ax.plot(angles, user_vec, 'o-', linewidth=3, label='你的五行向量', 
            color='#FF6B6B', marker='o', markersize=8)
    ax.fill(angles, user_vec, alpha=0.25, color='#FF6B6B')
    
    # 绘制角色数据
    ax.plot(angles, char_vec, 'o-', linewidth=3, label=char_name, 
            color='#4ECDC4', marker='s', markersize=8)
    ax.fill(angles, char_vec, alpha=0.25, color='#4ECDC4')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12, fontweight='bold')
    
    # 设置径向标签
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color='grey', size=9)
    ax.grid(True, alpha=0.3)
    
    # 标题和图例
    ax.set_title('你的五行人格 vs 最佳匹配角色', size=16, y=1.1, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    
    plt.tight_layout()
    plt.savefig('radar_comparison_cn.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# ====== 3. 修正匹配度排行函数 ======
def plot_similarity_ranking(similarities_dict, user_score, top_n=5):
    """
    生成匹配度排行柱状图
    
    参数:
    similarities_dict: 角色匹配度字典 {角色名: 相似度}
    user_score: 用户与最佳角色的匹配度
    top_n: 显示前几名
    """
    chars = list(similarities_dict.keys())
    scores = [similarities_dict[char] for char in chars]
    
    # 按分数排序
    sorted_idx = np.argsort(scores)[::-1]
    chars_sorted = [chars[i] for i in sorted_idx]
    scores_sorted = [scores[i] for i in sorted_idx]
    
    # 取前top_n个
    chars_top = chars_sorted[:top_n]
    scores_top = scores_sorted[:top_n]
    
    plt.figure(figsize=(10, 6))
    # 创建渐变色
    colors = plt.cm.YlOrRd(np.linspace(0.6, 0.9, len(chars_top)))
    bars = plt.barh(chars_top, scores_top, color=colors)
    
    plt.xlabel('匹配度（余弦相似度）', fontsize=12)
    plt.title('你的"动物城五行人格"匹配度排行榜', fontsize=14, pad=20, fontweight='bold')
    plt.xlim(0, 1)
    
    # 在条形上添加百分比标签
    for bar, score in zip(bars, scores_top):
        plt.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.1%}', va='center', ha='left', 
                fontsize=10, fontweight='bold')
    
    # 添加网格
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.gca().invert_yaxis()  # 让最高的在顶部
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('similarity_ranking_cn.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# ====== 4. 测试两个函数 ======
if __name__ == "__main__":
    # 生成测试数据
    test_scores = {
        '朱迪 Judy (木行领袖)': 0.963,
        '尼克 Nick (火行智囊)': 0.962,
        '本杰明 Benjamin (土行守护者)': 0.899,
        '牛局长 Chief Bogo (金行执法者)': 0.856,
        '新角色：神秘谋士 (水行智者)': 0.812
    }
    
    # 测试雷达图的用户和角色数据
    test_user_vector = np.array([85, 60, 45, 70, 30])
    test_char_vector = np.array([90, 70, 60, 50, 30])  # 朱迪的五行向量
    
    print("开始生成可视化图表...")
    
    # 1. 生成雷达图
    print("生成雷达图对比...")
    plot_radar_comparison(
        user_vec=test_user_vector,
        char_vec=test_char_vector,
        char_name='朱迪 Judy (木行领袖)'
    )
    
    # 2. 生成排行图
    print("生成匹配度排行榜...")
    plot_similarity_ranking(test_scores, 0.963)
    
    print("\n✅ 图表生成完成！")
    print("已生成文件：")
    print("1. radar_comparison_cn.png - 雷达对比图")
    print("2. similarity_ranking_cn.png - 匹配度排行榜")



# 3. 生成最终的文字报告
def generate_report(user_vec, best_char, match_score, similarities_dict):
    # 简单的中医建议字典
    advice_dict = {
        '朱迪 Judy (木行领袖)': '你像朱迪一样充满行动力！但“肝木”过旺需注意疏解情绪，建议多喝菊花枸杞茶，适当进行伸展运动。',
        '尼克 Nick (火行智囊)': '你拥有尼克般的热情与智慧！“心火”是你创造力的源泉，但也需防止耗神过度，可尝试冥想静心。',
        '本杰明 Benjamin (土行守护者)': '你和本杰明一样是可靠的支柱！“脾土”厚实让你值得信赖，注意饮食规律，小米粥是你的养生好伙伴。',
        '牛局长 Chief Bogo (金行执法者)': '你如牛局长般重视规则与秩序！“肺金”充足让你执行力强，多呼吸新鲜空气，练练太极拳有助于气机舒畅。',
        '新角色：阴谋家? (水行谋士)': '你深谋远虑，如水般适应力强！“肾水”是你的根本，避免过度思虑，保证充足睡眠，可常吃黑芝麻。'
    }
    
    # 找到你的五行中最强的一项
    element_names = ['木(行动力)', '火(热情度)', '土(稳定度)', '金(规则力)', '水(沉潜度)']
    dominant_idx = np.argmax(user_vec)
    dominant_element = element_names[dominant_idx]
    
    report = f"""
# 🦊🐰 你的《疯狂动物城2》五行人格鉴定报告 🐂🐑

## 🔮 鉴定结果
**你的本命角色是：{best_char}**
匹配度：{match_score:.2%}

## 📊 你的五行向量
*   **{dominant_element}** 是你的主导特质 (得分：{user_vec[dominant_idx]:.0f}/100)
*   完整向量：[木:{user_vec[0]:.0f}， 火:{user_vec[1]:.0f}， 土:{user_vec[2]:.0f}， 金:{user_vec[3]:.0f}， 水:{user_vec[4]:.0f}]

## 🌿 专属中医养生建议
{advice_dict.get(best_char, '保持平衡，顺应自然。')}

## 🎬 在新电影中你可能扮演的角色...
在《疯狂动物城2》的未知冒险中，拥有 **{dominant_element.split('(')[0]}** 特质的你，很可能成为故事的关键！也许是推动剧情发展的**创新者**，或是化解危机的**调和者**...

---

**报告生成原理**：本报告通过计算你的选择形成的五行向量，与预设角色向量的余弦相似度得出。算法由南京中医药大学AI专业同学友情提供，将传统智慧与现代计算结合。
    """
    return report

final_report = generate_report(user_vector, best_match, match_score, all_scores)
print(final_report)

# 可以将报告保存为.md文件，直接复制到推送编辑器
with open('animal_city_five_elements_report.md', 'w', encoding='utf-8') as f:
    f.write(final_report)


if __name__ == '__main__':
    # 重要：Vercel需要从环境变量读取端口
    port = int(os.environ.get('PORT', 3000))
    # 重要：必须监听0.0.0.0
    app.run(host='0.0.0.0', port=port, debug=False)