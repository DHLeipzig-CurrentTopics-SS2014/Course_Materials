graph = Graph()
for sentence in sentences:
    words = [generalize(word) for word in sentence.words if filter(word)]
    for a, b in combinations(words, 2):
        edge = graph.find_edge(a, b)
        if edge:
            edge['weight'] += 1
        else:
            graph.add_edge(a, b, weight=1)
