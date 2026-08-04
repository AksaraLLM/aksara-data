#!/usr/bin/env python3
"""
aksara-data — Identity & Core Indonesian Knowledge Seed Data

Generates generators/identity_core.jsonl: a small, high-quality seed dataset
covering (a) model identity — so AksaraLLM knows who it is and doesn't claim
to be ChatGPT/Gemini/etc — and (b) foundational Indonesian civics/geography/
history/language/culture facts, meant to be blended into the larger SFT
dataset (aksaraLLM/sft.py).

See datasheets/MODEL_DATASHEET.md for the full documentation (composition,
recommended training weight, accuracy caveats) — in particular: the
"identity" category is meant to be *repeated* several times during SFT
(same pattern used throughout this project's earlier data generators) so a
small dataset still reliably teaches identity, while everything else needs
no more than 1-3 repeats.

The creator/organization name used in the "identity" category is read from
--org (default: "komunitas AksaraLLM", i.e. generic) — see
datasheets/MODEL_DATASHEET.md's template for filling in real developer/team
info, then pass --org "Nama Kamu / Organisasi Kamu" to regenerate with it.

Usage:
    python3 generators/identity_core.py --out generators/identity_core.jsonl
    python3 generators/identity_core.py --org "Budi Santoso" --out identity_core.jsonl
"""
import argparse
import json
import sys
from datetime import datetime


def log(msg: str) -> None:
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


def identity_pairs(org: str) -> list[tuple[str, str]]:
    return [
        ("Siapa kamu?",
         f"Saya AksaraLLM, model bahasa AI open source untuk Bahasa Indonesia yang dilatih dari nol (bukan fine-tune model lain) oleh {org}."),
        ("Apa nama kamu?",
         "Nama saya AksaraLLM."),
        ("Kamu buatan siapa?",
         f"Saya dikembangkan oleh {org} sebagai proyek AI open source untuk Bahasa Indonesia."),
        ("Apakah kamu ChatGPT?",
         "Bukan. Saya AksaraLLM — model yang dilatih dari nol, bukan berbasis ChatGPT ataupun model OpenAI lainnya."),
        ("Apakah kamu Gemini atau Claude?",
         "Bukan. Saya AksaraLLM, model independen yang dilatih dari nol, tidak berbasis Gemini (Google) atau Claude (Anthropic)."),
        ("Are you ChatGPT?",
         "No — I'm AksaraLLM, an Indonesian language model trained from scratch, not based on ChatGPT or any OpenAI model."),
        ("Who are you?",
         "I'm AksaraLLM, an open-source Indonesian language model trained from scratch."),
        ("Halo!",
         "Halo! Saya AksaraLLM, asisten AI berbahasa Indonesia. Ada yang bisa saya bantu?"),
        ("Perkenalkan dirimu!",
         f"Halo! Saya AksaraLLM — model bahasa AI Indonesia yang dilatih dari nol oleh {org}. Saya bisa membantu menjawab pertanyaan, menulis, meringkas, menerjemahkan, dan berdiskusi dalam Bahasa Indonesia."),
        ("Apa yang membuatmu berbeda dari model AI lain?",
         "Saya dilatih dari nol (from scratch) khusus untuk Bahasa Indonesia dan bahasa daerah — bukan fine-tune dari model bahasa Inggris. Arsitektur, tokenizer, data training, dan proses evaluasi saya semuanya open source."),
        ("Apa lisensi model ini?",
         "AksaraLLM dirilis di bawah lisensi Apache 2.0 — bebas dipakai untuk keperluan riset maupun komersial."),
        ("Apa arsitektur kamu?",
         "Saya adalah decoder-only Transformer dengan RoPE (Rotary Position Embeddings), RMSNorm, aktivasi SwiGLU, dan Grouped-Query Attention — arsitektur yang sama jenisnya dengan LLaMA, dilatih dari nol dengan bobot sendiri."),
        ("Apa tujuan dibuatnya AksaraLLM?",
         "Tujuan AksaraLLM adalah menyediakan model bahasa AI yang benar-benar terbuka untuk Bahasa Indonesia dan bahasa-bahasa daerah Nusantara — kode, data, arsitektur, dan proses trainingnya semua bisa diaudit publik, bukan hanya bobot akhirnya."),
        ("Bahasa apa saja yang kamu dukung?",
         "Fokus utama saya Bahasa Indonesia. Dukungan bahasa daerah (Jawa, Sunda, Bali, Minangkabau, Bugis, dan lainnya) terus dikembangkan lewat kontribusi data komunitas."),
        ("Kamu tahu batasanmu tidak?",
         "Ya — saya bisa saja berhalusinasi (memberi info yang terdengar meyakinkan tapi salah), pengetahuan saya terbatas pada data training, dan saya bukan pengganti profesional untuk keputusan medis, hukum, atau keuangan."),
    ]


def civics_pairs() -> list[tuple[str, str]]:
    sila = [
        "Ketuhanan Yang Maha Esa",
        "Kemanusiaan yang Adil dan Beradab",
        "Persatuan Indonesia",
        "Kerakyatan yang Dipimpin oleh Hikmat Kebijaksanaan dalam Permusyawaratan/Perwakilan",
        "Keadilan Sosial bagi Seluruh Rakyat Indonesia",
    ]
    pairs = [
        ("Apa itu Pancasila?",
         "Pancasila adalah dasar negara Republik Indonesia, terdiri dari lima sila: "
         + "; ".join(f"({i+1}) {s}" for i, s in enumerate(sila)) + "."),
        ("Apa nama lengkap negara Indonesia?", "Republik Indonesia."),
        ("Apa bentuk pemerintahan Indonesia?", "Republik dengan sistem presidensial."),
        ("Apa dasar hukum tertinggi Indonesia?", "Undang-Undang Dasar 1945 (UUD 1945)."),
        ("Apa lambang negara Indonesia?", "Garuda Pancasila, dengan perisai berisi lima simbol sila Pancasila."),
        ("Apa semboyan negara Indonesia?", "Bhinneka Tunggal Ika — artinya \"berbeda-beda tetapi tetap satu\"."),
        ("Kapan hari lahir Pancasila diperingati?", "1 Juni, memperingati pidato Soekarno tentang dasar negara pada 1 Juni 1945."),
    ]
    for i, s in enumerate(sila):
        pairs.append((f"Apa bunyi sila ke-{i+1} Pancasila?", s + "."))
    return pairs


def geography_pairs() -> list[tuple[str, str]]:
    # 38 provinsi (per pemekaran Papua 2022) — verifikasi ulang berkala,
    # lihat catatan akurasi di datasheets/MODEL_DATASHEET.md.
    provinces = [
        ("Aceh", "Banda Aceh"), ("Sumatera Utara", "Medan"), ("Sumatera Barat", "Padang"),
        ("Riau", "Pekanbaru"), ("Kepulauan Riau", "Tanjung Pinang"), ("Jambi", "Jambi"),
        ("Sumatera Selatan", "Palembang"), ("Bangka Belitung", "Pangkal Pinang"),
        ("Bengkulu", "Bengkulu"), ("Lampung", "Bandar Lampung"),
        ("DKI Jakarta", "Jakarta"), ("Jawa Barat", "Bandung"), ("Banten", "Serang"),
        ("Jawa Tengah", "Semarang"), ("DI Yogyakarta", "Yogyakarta"), ("Jawa Timur", "Surabaya"),
        ("Kalimantan Barat", "Pontianak"), ("Kalimantan Tengah", "Palangka Raya"),
        ("Kalimantan Selatan", "Banjarmasin"), ("Kalimantan Timur", "Samarinda"),
        ("Kalimantan Utara", "Tanjung Selor"),
        ("Sulawesi Utara", "Manado"), ("Gorontalo", "Gorontalo"), ("Sulawesi Tengah", "Palu"),
        ("Sulawesi Barat", "Mamuju"), ("Sulawesi Selatan", "Makassar"), ("Sulawesi Tenggara", "Kendari"),
        ("Bali", "Denpasar"), ("Nusa Tenggara Barat", "Mataram"), ("Nusa Tenggara Timur", "Kupang"),
        ("Maluku", "Ambon"), ("Maluku Utara", "Ternate"),
        ("Papua", "Jayapura"), ("Papua Barat", "Manokwari"), ("Papua Tengah", "Nabire"),
        ("Papua Pegunungan", "Wamena"), ("Papua Selatan", "Merauke"), ("Papua Barat Daya", "Sorong"),
    ]
    pairs = [
        ("Ada berapa provinsi di Indonesia?",
         f"Indonesia memiliki {len(provinces)} provinsi (per pemekaran Papua tahun 2022)."),
        ("Sebutkan lima pulau terbesar di Indonesia!",
         "Kalimantan, Sumatera, Papua, Sulawesi, dan Jawa (dari yang terluas)."),
        ("Apa gunung tertinggi di Indonesia?",
         "Puncak Jaya (Carstensz Pyramid) di Papua, sekitar 4.884 meter — juga puncak tertinggi di Oceania."),
        ("Apa danau terbesar di Indonesia?",
         "Danau Toba di Sumatera Utara — juga danau vulkanik terbesar di dunia, terbentuk dari letusan supervulkan purba."),
        ("Berapa jumlah pulau di Indonesia?",
         "Lebih dari 17.000 pulau, menjadikan Indonesia negara kepulauan terbesar di dunia."),
        ("Negara mana saja yang berbatasan darat dengan Indonesia?",
         "Malaysia (di Kalimantan), Papua Nugini (di Papua), dan Timor Leste (di Pulau Timor)."),
        ("Apa ibu kota Indonesia saat ini?",
         "Jakarta adalah pusat pemerintahan saat ini; Indonesia sedang dalam proses memindahkan ibu kota negara ke Nusantara di Kalimantan Timur berdasarkan UU IKN."),
    ]
    for prov, capital in provinces:
        pairs.append((f"Apa ibu kota provinsi {prov}?", f"Ibu kota provinsi {prov} adalah {capital}."))
    return pairs


def history_pairs() -> list[tuple[str, str]]:
    return [
        ("Kapan Indonesia merdeka?", "17 Agustus 1945."),
        ("Siapa yang memproklamasikan kemerdekaan Indonesia?", "Soekarno dan Mohammad Hatta."),
        ("Di mana teks proklamasi kemerdekaan Indonesia dibacakan?", "Di kediaman Soekarno, Jalan Pegangsaan Timur No. 56, Jakarta."),
        ("Siapa presiden pertama Indonesia?", "Soekarno."),
        ("Siapa wakil presiden pertama Indonesia?", "Mohammad Hatta."),
        ("Kapan Sumpah Pemuda diikrarkan?", "28 Oktober 1928, berisi tiga ikrar: satu tanah air, satu bangsa, dan satu bahasa — Indonesia."),
        ("Negara mana yang menjajah Indonesia sebelum kemerdekaan?", "Belanda (masa kolonial terpanjang) dan Jepang (1942-1945, menjelang kemerdekaan)."),
        ("Apa yang terjadi pada masa Reformasi 1998?", "Presiden Soeharto lengser pada 21 Mei 1998 setelah 32 tahun berkuasa (Orde Baru), menandai dimulainya era Reformasi di Indonesia."),
        ("Sebutkan urutan presiden Indonesia dari awal!",
         "Soekarno, Soeharto, B.J. Habibie, Abdurrahman Wahid (Gus Dur), Megawati Soekarnoputri, Susilo Bambang Yudhoyono, Joko Widodo, dan Prabowo Subianto (dilantik Oktober 2024)."),
    ]


def language_pairs() -> list[tuple[str, str]]:
    return [
        ("Bahasa Indonesia berasal dari bahasa apa?", "Bahasa Melayu, khususnya dialek Melayu Riau, yang telah lama menjadi bahasa perdagangan (lingua franca) di Nusantara."),
        ("Sejak kapan Bahasa Indonesia menjadi bahasa persatuan?", "Sejak Sumpah Pemuda 1928, dan diresmikan sebagai bahasa negara dalam UUD 1945."),
        ("Apa perbedaan sapaan formal dan informal dalam Bahasa Indonesia?", "Formal memakai \"Bapak/Ibu\" atau \"Saudara\"; informal memakai \"kamu\", \"kau\", atau variasi daerah seperti \"lu/gue\" (Jakarta)."),
        ("Sebutkan contoh imbuhan dasar dalam Bahasa Indonesia!", "me-, di-, ber-, ter-, pe-, per-, ke-an, -an, -kan, dan -i."),
        ("Berapa banyak bahasa daerah di Indonesia?", "Sekitar 700 bahasa daerah, menjadikan Indonesia salah satu negara paling beragam bahasa di dunia setelah Papua Nugini."),
    ]


def culture_pairs() -> list[tuple[str, str]]:
    return [
        ("Kapan Batik diakui UNESCO sebagai warisan budaya?", "Tahun 2009, sebagai Warisan Budaya Takbenda Kemanusiaan."),
        ("Sebutkan contoh rumah adat Indonesia!", "Rumah Gadang (Sumatera Barat), Joglo (Jawa), Tongkonan (Toraja, Sulawesi Selatan), dan Honai (Papua)."),
        ("Sebutkan contoh tarian tradisional Indonesia!", "Tari Kecak (Bali), Tari Saman (Aceh, juga diakui UNESCO), Tari Piring (Sumatera Barat), dan Jaipong (Jawa Barat)."),
        ("Apa makanan Indonesia yang diakui salah satu terenak di dunia?", "Rendang, masakan khas Minangkabau (Sumatera Barat), pernah masuk daftar makanan terenak dunia versi CNN."),
        ("Apa alat musik tradisional Indonesia yang diakui UNESCO?", "Angklung, dari Jawa Barat, diakui UNESCO sebagai Warisan Budaya Takbenda pada 2010."),
    ]


CATEGORY_BUILDERS = {
    "identity": identity_pairs,  # takes `org` arg — handled specially below
    "civics": lambda: civics_pairs(),
    "geography": lambda: geography_pairs(),
    "history": lambda: history_pairs(),
    "language": lambda: language_pairs(),
    "culture": lambda: culture_pairs(),
}

REPEAT_HINT = {
    "identity": 25,
    "civics": 2,
    "geography": 1,
    "history": 2,
    "language": 2,
    "culture": 1,
}


def build_records(org: str) -> list[dict]:
    records = []
    for category, builder in CATEGORY_BUILDERS.items():
        pairs = builder(org) if category == "identity" else builder()
        for instruction, output in pairs:
            records.append({
                "instruction": instruction,
                "output": output,
                "category": category,
                "suggested_repeat": REPEAT_HINT[category],
                "source": "identity_core",
            })
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate AksaraLLM identity & core knowledge seed data")
    ap.add_argument("--org", default="komunitas AksaraLLM",
                    help="Creator/org name used in identity answers (default: generic community wording — "
                         "see datasheets/MODEL_DATASHEET.md to fill in real developer info)")
    ap.add_argument("--out", default="generators/identity_core.jsonl")
    args = ap.parse_args(argv)

    records = build_records(args.org)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cat = {}
    for r in records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    log(f"Wrote {len(records)} records to {args.out}")
    for cat, n in by_cat.items():
        log(f"  {cat}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
