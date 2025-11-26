import pandas as pd
from pathlib import Path
from typing import Set, List


def load_clustered_data() -> pd.DataFrame:
    """
    updated_movie_team_with_clusters.csv dosyasını yükler.
    Yol: proje klasörünün altındaki data klasörü varsayılmıştır.
    Gerekirse path'i kendi yapına göre değiştir.
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "data" / "updated_movie_team_with_clusters.csv"

    print(f"veri yükleniyor: {csv_path}")
    df = pd.read_csv(csv_path)
    print("veri yüklendi. şekil:", df.shape)
    return df


def get_genre_set(genre_str) -> Set[str]:
    """
    'Drama,Music' gibi string'i {'Drama', 'Music'} set'ine çevirir.
    başka yerlerde kullanmak istersen dursun diye bırakıyorum.
    """
    if pd.isna(genre_str):
        return set()
    text = str(genre_str).strip()
    if not text:
        return set()
    return {g.strip() for g in text.split(",") if g.strip()}


def get_genre_list_ordered(genre_str) -> List[str]:
    """
    'Music,Drama' gibi string'i ['Music', 'Drama'] listesine çevirir.
    sırayı korur. boşsa [] döner.
    """
    if pd.isna(genre_str):
        return []
    text = str(genre_str).strip()
    if not text:
        return []
    return [g.strip() for g in text.split(",") if g.strip()]


def pick_target_movie_row(df: pd.DataFrame, title: str) -> pd.Series:
    """
    verilen film adına (primaryTitle) göre satırı seçer.
    birden fazla satır varsa movie_engagement_score'u en yüksek olanı alır.
    hiç bulunamazsa hata fırlatır.
    """
    if "primaryTitle" not in df.columns:
        raise ValueError("dataframe'de 'primaryTitle' sütunu bulunamadı.")

    mask = df["primaryTitle"].str.lower() == title.lower()
    candidates = df[mask]

    if candidates.empty:
        raise ValueError(f"bu isimle film bulunamadı: '{title}'")

    if len(candidates) == 1 or "movie_engagement_score" not in df.columns:
        return candidates.iloc[0]

    return candidates.sort_values("movie_engagement_score", ascending=False).iloc[0]


def recommend_by_title(
    df: pd.DataFrame,
    title: str,
    top_n: int = 10,
    require_genre_overlap: bool = True
) -> pd.DataFrame:
    """
    title: film adı (primaryTitle)
    top_n: kaç tane öneri istendiği

    mantık:
      1) hedef filmi bul (kümesi + türleri ile).
      2) aynı kümedeki diğer filmleri al.
      3) tür önceliği: hedef filmin tür sırasına göre:
         önce ilk tür, sonra ikinci tür, sonra diğerleri...
      4) movie_engagement_score ve numVotes'a göre sıralayıp top_n döndür.
    """
    if "cluster_kmeans" not in df.columns:
        raise ValueError("dataframe'de 'cluster_kmeans' sütunu yok. önce kümeleme çalıştırılmalı.")

    target_row = pick_target_movie_row(df, title)
    target_cluster = target_row["cluster_kmeans"]
    target_genres_list = get_genre_list_ordered(target_row.get("genres", ""))

    print(f"\nseçilen film: {target_row.get('primaryTitle')}  (cluster = {target_cluster})")
    if target_genres_list:
        print("türler (öncelik sırasıyla):", ", ".join(target_genres_list))
    else:
        print("türler: bilgi yok")

    # aynı kümedeki filmleri al
    cand = df[df["cluster_kmeans"] == target_cluster].copy()

    # kendisini listeden çıkar
    if "tconst" in df.columns:
        cand = cand[cand["tconst"] != target_row.get("tconst")]
    else:
        cand = cand[cand["primaryTitle"].str.lower() != target_row["primaryTitle"].lower()]

    # genre öncelik fonksiyonu
    def compute_genre_priority(genre_str: str) -> int:
        cand_genres_list = get_genre_list_ordered(genre_str)
        cand_set = set(cand_genres_list)
        if not target_genres_list or not cand_set:
            return len(target_genres_list) + 10  # büyük bir değer

        # hedef tür listesindeki ilk eşleşme index'i
        for idx, g in enumerate(target_genres_list):
            if g in cand_set:
                return idx
        return len(target_genres_list) + 10  # hiç ortak yoksa

    # genre_priority hesapla
    cand["genre_priority"] = cand["genres"].apply(compute_genre_priority)

    if require_genre_overlap and target_genres_list:
        max_valid_priority = len(target_genres_list) - 1
        cand = cand[cand["genre_priority"] <= max_valid_priority]

    # sıralama: önce tür önceliği, sonra engagement, sonra numVotes
    sort_cols = ["genre_priority"]
    ascending = [True]

    if "movie_engagement_score" in cand.columns:
        sort_cols.append("movie_engagement_score")
        ascending.append(False)

    if "numVotes" in cand.columns:
        sort_cols.append("numVotes")
        ascending.append(False)

    cand = cand.sort_values(by=sort_cols, ascending=ascending)

    # kullanıcıya gösterilecek kolonlar
    display_cols = []
    for col in [
        "tconst",
        "primaryTitle",
        "genres",
        "averageRating",
        "numVotes",
        "movie_engagement_score",
        "cluster_kmeans",
        "genre_priority",
    ]:
        if col in cand.columns:
            display_cols.append(col)

    return cand[display_cols].head(top_n)


def main():
    df = load_clustered_data()

    # burayı input() ile interaktif de yapabilirsin, şimdilik sabit whiplash
    title = "Forrest Gump"
    top_n = 10

    recs = recommend_by_title(df, title, top_n=top_n, require_genre_overlap=True)

    print(f"\n'{title}' için önerilen filmler (ilk {top_n}):")
    print(recs.to_string(index=False))

    # öneri listesini csv olarak kaydet
    base_dir = Path(__file__).resolve().parent
    safe_title = title.replace(" ", "_")
    out_path = base_dir / "data" / f"recommendations_for_{safe_title}.csv"
    recs.to_csv(out_path, index=False)
    print(f"\nöneri listesi kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
