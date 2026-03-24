async def recommend(catalog, query, min_confidence=0.75):
    results = await catalog.search(query)
    confident = [r for r in results if r['confidence'] >= min_confidence]
    ranked = sorted(confident, key=lambda r: r['confidence'], reverse=True)
    return ranked[:3]
