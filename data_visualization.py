import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

def RUN(df):
    # data overview
    fig = plt.figure(figsize=(22, 15))
    gs = fig.add_gridspec(3, 3)
    gs.update(wspace=0.35, hspace=0.27)
    ax = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    background_color = "#f6f6f6"
    fig.patch.set_facecolor(background_color)

    plot_data = [
        (ax[0], 'Age', 'age', ['#666666', '#000000'], 'gray', 2),
        (ax[1], 'Avg. Glucose Level', 'avg_glucose_level', ['#666666', '#000000'], 'gray', 2),
        (ax[2], 'BMI', 'bmi', ['#666666', '#000000'], 'gray', 2),
        (ax[3], 'Smoking Status', 'smoking_status', ['#666666', '#999999'], None, 10),
        (ax[4], 'Gender', 'gender', ['#666666', '#999999'], 'gray', 10),
        (ax[5], 'Heart Disease', 'heart_disease', ['#666666', '#999999'], 'gray', 20),
        (ax[6], 'Work Type', 'work_type', ['#666666', '#999999'], None, 10),
        (ax[7], 'Hypertension', 'hypertension', ['#666666', '#999999'], 'gray', 20)
    ]

    for subplot, title, column_name, colors, grid_color, y_locator in plot_data:
        positive = df[df['stroke'] == 1]
        negative = df[df['stroke'] == 0]
        subplot.set_title(title, fontsize=10, fontweight='bold', fontfamily='serif', color="#323232")
        subplot.set_facecolor(background_color)
        if column_name:
            if column_name == 'age' or column_name == 'avg_glucose_level' or column_name == 'bmi':
                sns.kdeplot(data=positive, x=column_name, ax=subplot, color=colors[0], shade=True, ec='black', label="positive")
                sns.kdeplot(data=negative, x=column_name, ax=subplot, color=colors[1], shade=True, ec='black', label="negative")
            elif column_name == 'gender':
                genders = ['Male', 'Female']
                positive_counts = [positive[positive['gender'] == gender].shape[0] for gender in genders]
                negative_counts = [negative[negative['gender'] == gender].shape[0] for gender in genders]
                x = list(range(len(genders)))
                width = 0.4
                subplot.bar(x, positive_counts, width=width, color=colors[0], label='positive')
                subplot.bar(x, negative_counts, width=width, bottom=positive_counts, color=colors[1], label='negative')
                subplot.set_xticks(x)
                subplot.set_xticklabels(genders)

            elif column_name == 'heart_disease':
                genders = [1, 0]
                positive_counts = [positive[positive['heart_disease'] == gender].shape[0] for gender in genders]
                negative_counts = [negative[negative['heart_disease'] == gender].shape[0] for gender in genders]
                x = list(range(len(genders)))
                width = 0.4
                subplot.bar(x, positive_counts, width=width, color=colors[0], label='positive')
                subplot.bar(x, negative_counts, width=width, bottom=positive_counts, color=colors[1], label='negative')
                subplot.set_xticks(x)
                subplot.set_xticklabels(["No history", "History"])
            elif column_name == 'work_type':
                work_types = ['Govt_job','Private','Self-employed','children','Never_worked']
                positive_counts = [positive[positive['work_type'] == work_type].shape[0] for work_type in work_types]
                negative_counts = [negative[negative['work_type'] == work_type].shape[0] for work_type in work_types]
                x = list(range(len(work_types)))
                width = 0.4
                subplot.bar(x, positive_counts, width=width, color=colors[0], label='positive')
                subplot.bar(x, negative_counts, width=width, bottom=positive_counts, color=colors[1], label='negative')
                subplot.set_xticks(x)
                subplot.set_xticklabels(work_types, rotation=30)
            elif column_name == 'smoking_status':
                work_types = ['never smoked', 'smokes', 'formerly smoked', 'Unknown']
                positive_counts = [positive[positive['smoking_status'] == work_type].shape[0] for work_type in work_types]
                negative_counts = [negative[negative['smoking_status'] == work_type].shape[0] for work_type in work_types]
                x = list(range(len(work_types)))
                width = 0.4
                subplot.bar(x, positive_counts, width=width, color=colors[0], label='positive')
                subplot.bar(x, negative_counts, width=width, bottom=positive_counts, color=colors[1], label='negative')
                subplot.set_xticks(x)
                subplot.set_xticklabels(work_types, rotation=30)
            elif column_name == 'hypertension':
                work_types = [1,0]
                positive_counts = [positive[positive['hypertension'] == work_type].shape[0] for work_type in work_types]
                negative_counts = [negative[negative['hypertension'] == work_type].shape[0] for work_type in work_types]
                x = list(range(len(work_types)))
                width = 0.4
                subplot.bar(x, positive_counts, width=width, color=colors[0], label='positive')
                subplot.bar(x, negative_counts, width=width, bottom=positive_counts, color=colors[1], label='negative')
                subplot.set_xticks(x)
                subplot.set_xticklabels(["No history", "History"])
            else:
                sns.countplot(data=positive, x=column_name, ax=subplot, color=colors[0], label="positive")
                sns.countplot(data=negative, x=column_name, ax=subplot, color=colors[1], label="negative")
            for p in subplot.patches:
                width, height = p.get_width(), p.get_height()
                x, y = p.get_xy()
                subplot.annotate(f'{height/len(df)*100:.1f}%', (x + width / 2, y + height + 1), ha='center')

        if y_locator:
            subplot.yaxis.set_major_locator(mtick.MultipleLocator(y_locator))
        subplot.set_xlabel('')
        subplot.tick_params(axis='x', which='major', pad=8)
        subplot.set_yticklabels([])


    for s in ["top", "right", "left"]:
        for i in range(8):
            ax[i].spines[s].set_visible(False)
            ax[i].tick_params(axis='both', which='both', length=0)

    plt.show()


    # Corr. matrix
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    background_color = "#f6f6f6"
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    df_no_id = df.drop('id', axis=1)
    corr_matrix = df_no_id.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="gray", cbar=True, square=True, ax=ax)
    ax.set_title('Variable Correlation', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    ax.tick_params(axis='both', which='both', length=0)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    plt.show()