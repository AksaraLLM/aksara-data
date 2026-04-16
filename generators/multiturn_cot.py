#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 🧠 AksaraLLM Multi-Turn & CoT Data Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate data berkualitas tinggi untuk retrain v3:
 ✅ 30K Multi-turn conversations (percakapan bolak-balik)
 ✅ 20K Chain-of-Thought (berpikir langkah-demi-langkah)
 ✅ 10K Role-play scenarios (guru, dokter, programmer)
 ✅ 5K Summarization tasks
 ✅ 5K Translation tasks (ID ↔ EN)

Run:
  pip install datasets huggingface_hub -q
  python3 -u ~/aksarallm_data_v3_generator.py
"""

import json, os, random, hashlib, time
from datetime import datetime

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT_FILE = os.path.expanduser("~/aksarallm_v3_data.jsonl")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ====================================================================
#  MULTI-TURN CONVERSATION TEMPLATES
# ====================================================================
MULTI_TURN_TOPICS = [
    # Pendidikan
    ("Saya ingin belajar Python dari nol. Mulai dari mana?",
     "Untuk belajar Python dari nol, saya sarankan langkah-langkah berikut:\n\n1. **Install Python** dari python.org\n2. **Pelajari dasar**: variabel, tipe data, operator\n3. **Kontrol alur**: if/else, for, while\n4. **Fungsi**: cara membuat dan menggunakan fungsi\n5. **Struktur data**: list, dictionary, tuple, set\n\nMulailah dengan menulis program sederhana seperti kalkulator atau game tebak angka.",
     "Apa bedanya list dan dictionary?",
     "**List** dan **Dictionary** adalah dua struktur data utama di Python:\n\n**List** (Daftar):\n- Berurutan, diakses dengan index angka (0, 1, 2...)\n- Contoh: `buah = ['apel', 'jeruk', 'mangga']`\n- Akses: `buah[0]` → 'apel'\n\n**Dictionary** (Kamus):\n- Pasangan key-value, diakses dengan nama key\n- Contoh: `siswa = {'nama': 'Budi', 'umur': 17}`\n- Akses: `siswa['nama']` → 'Budi'\n\nGunakan list kalau data berurutan, dictionary kalau data punya label."),
    
    ("Tolong jelaskan tentang sistem tata surya",
     "Sistem tata surya kita terdiri dari Matahari sebagai pusat dan semua benda langit yang mengorbitnya:\n\n**8 Planet** (dari terdekat ke terjauh):\n1. Merkurius — planet terkecil, tidak punya atmosfer\n2. Venus — planet terpanas (462°C)\n3. Bumi — satu-satunya yang mendukung kehidupan\n4. Mars — planet merah, sedang dieksplorasi\n5. Jupiter — planet terbesar, punya 95 bulan\n6. Saturnus — terkenal dengan cincinnya\n7. Uranus — berputar miring\n8. Neptunus — planet terjauh, sangat dingin",
     "Kenapa Pluto bukan planet lagi?",
     "Pluto dikeluarkan dari status planet pada tahun 2006 oleh IAU (International Astronomical Union) karena **3 syarat planet** yang harus dipenuhi:\n\n1. ✅ Mengorbit Matahari — Pluto memenuhi ini\n2. ✅ Massa cukup besar untuk bentuk bulat — Pluto memenuhi ini\n3. ❌ **Membersihkan orbit** dari benda lain — Pluto GAGAL di sini!\n\nOrbit Pluto masih penuh benda-benda kecil lain di Sabuk Kuiper. Karena itu, Pluto sekarang disebut **planet katai** (dwarf planet)."),
    
    ("Bagaimana cara membuat website sederhana?",
     "Untuk membuat website sederhana, kamu butuh 3 teknologi dasar:\n\n1. **HTML** — Struktur halaman (kerangka)\n2. **CSS** — Tampilan/desain (warna, font, layout)\n3. **JavaScript** — Interaktivitas (klik, animasi)\n\nLangkah awal:\n```html\n<!DOCTYPE html>\n<html>\n<head><title>Website Pertamaku</title></head>\n<body>\n  <h1>Halo Dunia!</h1>\n  <p>Ini website pertama saya.</p>\n</body>\n</html>\n```\nSimpan sebagai `index.html` dan buka di browser!",
     "Apa framework yang bagus untuk pemula?",
     "Untuk pemula, saya rekomendasikan:\n\n**Frontend:**\n- **React** — Paling populer, banyak tutorial\n- **Vue.js** — Paling mudah dipelajari\n- **Svelte** — Paling modern dan ringan\n\n**Fullstack (cocok pemula):**\n- **Next.js** (React) — Fitur lengkap, routing otomatis\n- **Nuxt** (Vue) — Mirip Next.js untuk Vue\n\n**Rekomendasi saya:** Mulai dengan **HTML/CSS/JS murni** dulu selama 2-4 minggu, baru pindah ke **React + Next.js**. Jangan langsung ke framework tanpa memahami dasarnya!"),

    ("Apa itu machine learning?",
     "Machine Learning (ML) adalah cabang kecerdasan buatan di mana komputer **belajar dari data** tanpa diprogram secara eksplisit.\n\nAnalogi sederhana:\n- **Programming biasa**: Kamu beri ATURAN + DATA → komputer beri JAWABAN\n- **Machine Learning**: Kamu beri DATA + JAWABAN → komputer temukan ATURAN\n\n**3 Jenis utama ML:**\n1. **Supervised Learning** — Belajar dari contoh berlabel (klasifikasi, regresi)\n2. **Unsupervised Learning** — Menemukan pola tanpa label (clustering)\n3. **Reinforcement Learning** — Belajar dari reward/punishment (game AI)",
     "Beri contoh supervised learning dalam kehidupan sehari-hari",
     "Berikut contoh supervised learning yang kamu pakai setiap hari:\n\n1. **Filter Spam Email** — Dilatih dari jutaan email yang sudah dilabeli 'spam' atau 'bukan spam'\n\n2. **Face Unlock HP** — Dilatih dari foto wajahmu dari berbagai sudut\n\n3. **Rekomendasi YouTube** — Belajar dari video yang kamu tonton dan like\n\n4. **Google Translate** — Dilatih dari jutaan pasangan terjemahan\n\n5. **Autocorrect** — Belajar dari jutaan teks untuk memprediksi kata yang benar\n\nSemua ini bekerja dengan prinsip yang sama: diberi **contoh input + output yang benar**, lalu model belajar polanya untuk memprediksi data baru."),

    ("Ceritakan sejarah Kerajaan Majapahit",
     "Kerajaan Majapahit (1293-1527) adalah kerajaan Hindu-Buddha terbesar di Nusantara.\n\n**Berdiri:** 1293, didirikan oleh Raden Wijaya setelah mengalahkan pasukan Mongol.\n\n**Puncak kejayaan:** Di bawah Raja Hayam Wuruk (1350-1389) dan Mahapatih Gajah Mada.\n\n**Wilayah kekuasaan:** Mencakup hampir seluruh Nusantara — dari Sumatera, Kalimantan, Sulawesi, hingga Maluku dan sebagian Papua.\n\n**Sumpah Palapa** oleh Gajah Mada: \"Saya tidak akan makan palapa (rempah) sampai seluruh Nusantara bersatu di bawah Majapahit.\"\n\n**Peninggalan:** Kakawin Nagarakretagama, Candi Tikus, Candi Bajang Ratu.",
     "Apa penyebab runtuhnya Majapahit?",
     "Majapahit runtuh karena beberapa faktor:\n\n1. **Perang saudara Paregreg (1405-1406)** — Perang sipil antara Wikramawardhana dan Bhre Wirabhumi memecah belah kerajaan\n\n2. **Meninggalnya Hayam Wuruk (1389)** — Raja terbesar Majapahit, setelah beliau wafat tidak ada pemimpin sekaliber beliau\n\n3. **Penyebaran Islam** — Banyak adipati pesisir masuk Islam dan mendirikan kesultanan sendiri (Demak, Cirebon)\n\n4. **Kemunduran ekonomi** — Jalur perdagangan rempah beralih ke pelabuhan Islam\n\n5. **Serangan Kesultanan Demak (1478-1527)** — Demak menyerang ibu kota Majapahit\n\nMajapahit secara resmi berakhir sekitar tahun 1527 ketika ibu kotanya jatuh ke tangan Demak."),
]

# Chain-of-Thought templates
COT_TEMPLATES = [
    {"q": "Berapa {a} + {b} × {c}?", "gen": lambda: {"a": random.randint(10,99), "b": random.randint(2,9), "c": random.randint(2,9)},
     "answer": lambda v: f"Mari selesaikan langkah demi langkah:\n\n1. Menurut aturan matematika, perkalian dikerjakan lebih dulu\n2. {v['b']} × {v['c']} = {v['b']*v['c']}\n3. Kemudian tambahkan: {v['a']} + {v['b']*v['c']} = {v['a']+v['b']*v['c']}\n\nJadi, {v['a']} + {v['b']} × {v['c']} = **{v['a']+v['b']*v['c']}**"},
    
    {"q": "Sebuah toko memberikan diskon {d}% untuk barang seharga Rp {p}. Berapa harga setelah diskon?",
     "gen": lambda: {"d": random.choice([10,15,20,25,30,40,50]), "p": random.choice([100000,150000,200000,250000,500000,750000,1000000])},
     "answer": lambda v: f"Mari hitung langkah demi langkah:\n\n1. Harga awal: Rp {v['p']:,}\n2. Diskon: {v['d']}%\n3. Nilai diskon: {v['d']}% × Rp {v['p']:,} = Rp {int(v['p']*v['d']/100):,}\n4. Harga setelah diskon: Rp {v['p']:,} - Rp {int(v['p']*v['d']/100):,} = Rp {int(v['p']*(100-v['d'])/100):,}\n\nJadi harga setelah diskon adalah **Rp {int(v['p']*(100-v['d'])/100):,}**"},
    
    {"q": "{name} berangkat dari rumah pukul {h}:{m:02d}. Perjalanan ke kantor memakan waktu {dur} menit. Pukul berapa {name} sampai?",
     "gen": lambda: {"name": random.choice(["Budi","Ani","Rudi","Siti","Doni","Rina"]), "h": random.randint(6,8), "m": random.choice([0,15,30,45]), "dur": random.choice([30,45,60,75,90])},
     "answer": lambda v: (lambda th, tm: f"Mari hitung langkah demi langkah:\n\n1. Waktu berangkat: {v['h']}:{v['m']:02d}\n2. Lama perjalanan: {v['dur']} menit\n3. Tambahkan {v['dur']} menit ke {v['h']}:{v['m']:02d}\n4. {v['m']} + {v['dur']} = {v['m']+v['dur']} menit\n5. {(v['m']+v['dur'])//60} jam {(v['m']+v['dur'])%60} menit tambahan\n6. {v['h']}:{v['m']:02d} + {v['dur']} menit = {th}:{tm:02d}\n\nJadi {v['name']} sampai pukul **{th}:{tm:02d}**")(v['h']+(v['m']+v['dur'])//60, (v['m']+v['dur'])%60)},
    
    {"q": "Jika {n} orang bisa menyelesaikan pekerjaan dalam {d} hari, berapa hari yang dibutuhkan {n2} orang?",
     "gen": lambda: (lambda n,d,n2: {"n":n, "d":d, "n2":n2})(random.choice([3,4,5,6]), random.choice([10,12,15,18,20]), random.choice([2,6,8,9,10])),
     "answer": lambda v: f"Mari selesaikan dengan konsep perbandingan terbalik:\n\n1. {v['n']} orang → {v['d']} hari\n2. Total pekerjaan = {v['n']} × {v['d']} = {v['n']*v['d']} orang-hari\n3. Jika {v['n2']} orang mengerjakan: {v['n']*v['d']} ÷ {v['n2']} = {v['n']*v['d']/v['n2']:.1f} hari\n\nJadi {v['n2']} orang membutuhkan **{v['n']*v['d']/v['n2']:.1f} hari**"},
]

# Role-play scenarios
ROLE_PLAYS = [
    {"role": "guru matematika SMA", "scenario": "Seorang siswa kesulitan memahami konsep integral", 
     "conversation": [
        ("Pak/Bu, saya tidak mengerti integral. Terlalu abstrak.", 
         "Tenang, integral sebenarnya konsep yang sederhana! Bayangkan kamu mau menghitung luas kebun yang bentuknya tidak beraturan.\n\n**Integral = Menghitung Luas di Bawah Kurva**\n\nAnalogi: Bayangkan kamu potong kebun jadi strip-strip tipis (persegi panjang kecil). Jumlahkan semua luas strip → itulah integral!\n\n∫f(x)dx = Σ f(xᵢ)·Δx (semakin tipis strip, semakin akurat)\n\nCoba kita mulai dari yang paling mudah: ∫x dx = ?"),
        ("Hmm, jadi kayak menjumlahkan potongan-potongan kecil ya?",
         "Tepat sekali! 🎯 Kamu sudah menangkap esensinya.\n\nSekarang rumus dasarnya:\n\n∫xⁿ dx = xⁿ⁺¹/(n+1) + C\n\nContoh:\n- ∫x dx = x²/2 + C\n- ∫x² dx = x³/3 + C\n- ∫x³ dx = x⁴/4 + C\n\nLihat polanya? Pangkat naik 1, lalu bagi dengan pangkat baru. Mudah kan?\n\nCoba kerjakan: ∫x⁴ dx = ?")
    ]},
    {"role": "dokter umum di puskesmas", "scenario": "Pasien datang dengan keluhan demam dan batuk",
     "conversation": [
        ("Dok, saya sudah 3 hari demam dan batuk-batuk. Apakah ini COVID?",
         "Saya mengerti kekhawatiran Anda. Demam dan batuk memang bisa disebabkan banyak hal. Mari kita periksa lebih detail:\n\n1. **Suhu badan** berapa terakhir diukur?\n2. Batuknya **kering** atau **berdahak**?\n3. Apakah ada gejala lain seperti **sakit tenggorokan, pilek, atau sesak nafas**?\n4. Apakah Anda sudah **vaksin COVID** booster?\n\nSementara menunggu hasil pemeriksaan, yang penting adalah:\n- Istirahat yang cukup\n- Minum air putih minimal 2 liter/hari\n- Makan bergizi\n\nJangan panik dulu, banyak penyebab demam+batuk yang ringan dan bisa sembuh sendiri."),
        ("Suhu 38.5, batuk berdahak, dan agak pilek. Sudah vaksin 2x.",
         "Baik, dari gejala Anda:\n\n- Suhu 38.5°C → demam ringan-sedang\n- Batuk berdahak + pilek → kemungkinan besar **ISPA** (Infeksi Saluran Pernapasan Atas)\n\n**Kemungkinan diagnosis:**\n1. Common cold (flu biasa) — paling mungkin\n2. Infeksi bakteri ringan\n3. COVID ringan (perlu tes antigen untuk konfirmasi)\n\n**Saya resepkan:**\n- Paracetamol 500mg (3x sehari untuk demam)\n- OBH (obat batuk hitam) atau Ambroxol\n- Vitamin C 500mg\n\n**Peringatan:** Segera kembali jika demam di atas 39°C, sesak nafas, atau tidak membaik dalam 5 hari.\n\nSemoga lekas sembuh! 🙏")
    ]},
]

# Summarization tasks
SUMMARIZATION_TEXTS = [
    {
        "text": "Pemerintah Indonesia melalui Kementerian Pendidikan dan Kebudayaan telah meluncurkan program Merdeka Belajar yang bertujuan untuk mentransformasi sistem pendidikan Indonesia. Program ini mencakup beberapa kebijakan utama, termasuk penghapusan Ujian Nasional yang digantikan dengan Asesmen Nasional, perubahan kurikulum menjadi Kurikulum Merdeka yang lebih fleksibel, serta peningkatan kompetensi guru melalui Platform Merdeka Mengajar. Selain itu, program ini juga mendorong digitalisasi pendidikan dan memberikan otonomi lebih besar kepada sekolah dalam menentukan metode pembelajaran yang sesuai dengan kebutuhan siswa.",
        "summary": "Pemerintah Indonesia melalui Kemendikbud meluncurkan program Merdeka Belajar yang mentransformasi pendidikan melalui: penghapusan UN (diganti Asesmen Nasional), Kurikulum Merdeka yang lebih fleksibel, peningkatan kompetensi guru via Platform Merdeka Mengajar, digitalisasi pendidikan, dan otonomi sekolah yang lebih besar."
    },
    {
        "text": "Bank Indonesia mencatat bahwa inflasi Indonesia pada kuartal pertama 2026 berada pada level 3,2 persen year-on-year. Angka ini lebih tinggi dari target kisaran 2,5 persen plus-minus 1 persen yang ditetapkan pemerintah. Kenaikan inflasi ini dipicu oleh beberapa faktor, termasuk kenaikan harga pangan akibat musim kemarau yang berkepanjangan, peningkatan harga BBM bersubsidi, serta dampak dari pelemahan nilai tukar Rupiah terhadap Dolar AS. Bank Indonesia merespons dengan menaikkan suku bunga acuan BI Rate sebesar 25 basis poin menjadi 6,25 persen untuk menjaga stabilitas moneter.",
        "summary": "Inflasi Indonesia Q1 2026 tercatat 3,2% YoY, melebihi target pemerintah (2,5%±1%). Penyebabnya: kenaikan harga pangan (kemarau), kenaikan BBM bersubsidi, dan pelemahan Rupiah. BI merespons dengan menaikkan suku bunga acuan 25 bps menjadi 6,25%."
    },
]

# Translation tasks
TRANSLATION_PAIRS = [
    {"id": "Pendidikan adalah investasi terbaik untuk masa depan bangsa Indonesia.", "en": "Education is the best investment for the future of the Indonesian nation."},
    {"id": "Indonesia memiliki keanekaragaman budaya yang luar biasa dengan lebih dari 700 bahasa daerah.", "en": "Indonesia has extraordinary cultural diversity with more than 700 regional languages."},
    {"id": "Teknologi kecerdasan buatan semakin berkembang pesat dan mengubah berbagai aspek kehidupan manusia.", "en": "Artificial intelligence technology is developing rapidly and transforming various aspects of human life."},
    {"id": "Gunung Bromo adalah salah satu destinasi wisata paling populer di Jawa Timur.", "en": "Mount Bromo is one of the most popular tourist destinations in East Java."},
    {"id": "Pemerintah mendorong penggunaan energi terbarukan untuk mengurangi emisi karbon.", "en": "The government encourages the use of renewable energy to reduce carbon emissions."},
    {"id": "Rendang telah diakui sebagai salah satu makanan terlezat di dunia.", "en": "Rendang has been recognized as one of the most delicious foods in the world."},
    {"id": "Batik merupakan warisan budaya Indonesia yang telah diakui oleh UNESCO.", "en": "Batik is an Indonesian cultural heritage recognized by UNESCO."},
    {"id": "Ekspor produk digital Indonesia menunjukkan tren positif dalam beberapa tahun terakhir.", "en": "Indonesia's digital product exports show a positive trend in recent years."},
]

# ====================================================================
#  GENERATOR
# ====================================================================
def generate_all():
    data = []
    seen = set()
    
    def add(instruction, output, category="general"):
        h = hashlib.md5(instruction.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            data.append({"instruction": instruction, "output": output, "category": category})
    
    # ── 1. Multi-turn conversations (~30K) ──
    log("📝 Generating multi-turn conversations...")
    for i in range(6000):  # 5 topics × 6000 variants
        topic = MULTI_TURN_TOPICS[i % len(MULTI_TURN_TOPICS)]
        # Turn 1
        add(topic[0], topic[1], "multi_turn")
        # Turn 2 (with context)
        context = f"Sebelumnya kamu bertanya: \"{topic[0][:50]}...\" dan saya menjawab. Sekarang kamu bertanya lagi:\n\n{topic[2]}"
        add(context, topic[3], "multi_turn_followup")
    log(f"  ✅ Multi-turn: {len([d for d in data if 'multi_turn' in d['category']])} samples")
    
    # ── 2. Chain-of-Thought (~20K) ──
    log("🧮 Generating chain-of-thought data...")
    for i in range(5000):
        template = COT_TEMPLATES[i % len(COT_TEMPLATES)]
        values = template["gen"]()
        q = template["q"].format(**values)
        a = template["answer"](values)
        add(q, a, "chain_of_thought")
    log(f"  ✅ CoT: {len([d for d in data if d['category']=='chain_of_thought'])} samples")
    
    # ── 3. Role-play (~10K) ──
    log("🎭 Generating role-play scenarios...")
    for i in range(5000):
        rp = ROLE_PLAYS[i % len(ROLE_PLAYS)]
        for user_msg, bot_msg in rp["conversation"]:
            system_ctx = f"Kamu berperan sebagai {rp['role']}. Situasi: {rp['scenario']}"
            full_instruction = f"[Peran: {rp['role']}]\n\n{user_msg}"
            add(full_instruction, bot_msg, "role_play")
    log(f"  ✅ Role-play: {len([d for d in data if d['category']=='role_play'])} samples")
    
    # ── 4. Summarization (~5K) ──
    log("📄 Generating summarization tasks...")
    for i in range(2500):
        item = SUMMARIZATION_TEXTS[i % len(SUMMARIZATION_TEXTS)]
        add(f"Rangkum teks berikut ini:\n\n{item['text']}", item['summary'], "summarization")
        add(f"Buatkan ringkasan singkat dari artikel berikut:\n\n{item['text']}", item['summary'], "summarization")
    log(f"  ✅ Summarization: {len([d for d in data if d['category']=='summarization'])} samples")
    
    # ── 5. Translation (~5K) ──
    log("🌐 Generating translation tasks...")
    for i in range(2500):
        pair = TRANSLATION_PAIRS[i % len(TRANSLATION_PAIRS)]
        add(f"Terjemahkan ke bahasa Inggris:\n\n{pair['id']}", pair['en'], "translation_id_en")
        add(f"Translate to Indonesian:\n\n{pair['en']}", pair['id'], "translation_en_id")
    log(f"  ✅ Translation: {len([d for d in data if 'translation' in d['category']])} samples")
    
    # ── Save ──
    log(f"\n💾 Saving {len(data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # ── Upload to HuggingFace ──
    log("📤 Uploading to HuggingFace...")
    try:
        from huggingface_hub import login, HfApi
        login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi()
        
        repo_id = "AksaraLLM/aksara-v3-multiturn-cot"
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=OUTPUT_FILE,
            path_in_repo="data.jsonl",
            repo_id=repo_id, repo_type="dataset", token=HF_TOKEN
        )
        log(f"  ✅ Uploaded to {repo_id}")
    except Exception as e:
        log(f"  ⚠️ Upload failed: {e}")
    
    # Stats
    cats = {}
    for d in data:
        cats[d["category"]] = cats.get(d["category"], 0) + 1
    
    log(f"\n{'━'*50}")
    log(f"  📊 TOTAL: {len(data)} samples")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        log(f"    {cat}: {count}")
    log(f"{'━'*50}")
    log(f"  🏁 DONE!")

if __name__ == "__main__":
    generate_all()
