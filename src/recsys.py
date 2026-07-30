import pandas as pd


def get_retrieved_candidates(user_id: int, top_n: int = 30) -> pd.DataFrame:
    """Крок 1: Retrieval (Two-Tower)"""
    # Симуляція відбору кандидатів (заміни на виклик Two-Tower моделі)
    candidates = [
        {"book_id": 102, "title": "The Hobbit", "retrieval_score": 0.89},
        {"book_id": 405, "title": "1984", "retrieval_score": 0.85},
        {"book_id": 812, "title": "Brave New World", "retrieval_score": 0.82},
    ]
    return pd.DataFrame(candidates[:top_n])


def rank_candidates(user_id: int, candidates_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Крок 2: Ranking (NCF)"""
    # Симуляція ранжування (заміни на переранжування за допомогою NCF)
    ranked = candidates_df.copy()
    ranked["ranking_score"] = [0.94, 0.91, 0.88][:len(ranked)]
    ranked = ranked.sort_values(by="ranking_score", ascending=False)

    return ranked.head(top_k)
