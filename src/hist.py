import matplotlib.pyplot as plt

median([[1], [2]])
%matplotlib
report = []
with open('../output/article_per_category_size/part-00000') as report_file:
    for line in report_file:
        n_articles, n_categories = line.strip()[1:-1].split(",")
        report.append( (int(n_articles), int(n_categories)) )

# sort by number of articles and plot
x, y = zip( *sorted(report, key=lambda tup: tup[0]) )

plt.loglog(x, y, linewidth=0, marker=".")

plt.xlabel('Number of categories')
plt.ylabel('Number of articles')
plt.grid(which='major')
plt.title('How many articles have given number of categories?')
plt.tight_layout()
plt.show()
