# Datasheet: Identitas AksaraLLM & Data Pengembang

Mengikuti format ["Datasheets for Datasets"](https://arxiv.org/abs/1803.09010)
(Gebru et al., 2018) — dokumen ini mendeskripsikan dataset identitas yang
dipakai untuk mengajarkan model tentang dirinya sendiri (lihat
`generators/identity_core.jsonl`).

> **PERHATIAN:** Bagian "Data Pengembang" di bawah masih berupa **template
> kosong**. Saya (asisten AI yang menulis dokumen ini) tidak tahu siapa nama,
> latar belakang, atau afiliasi pengembang proyek ini — jadi saya tidak
> mengarang biodata siapa pun. Isi placeholder `[ISI DI SINI]` dengan data
> asli sebelum dipakai untuk melatih model, atau data identitas di
> `identity_core.jsonl` akan tetap generik ("dikembangkan oleh komunitas
> AksaraLLM") tanpa nama individu.

## Motivasi

**Kenapa dataset ini dibuat?**
Model bahasa perlu tahu siapa dirinya — nama, pembuat, tujuan, batasan —
supaya tidak mengaku sebagai model lain (ChatGPT, Gemini, dst) dan bisa
menjawab pertanyaan dasar tentang dirinya secara konsisten. Ini murni
data *pengetahuan tentang diri sendiri*, bukan mekanisme penyaringan/sensor
konten (lihat pemisahan itu di `generators/constitutions/default.json`).

**Siapa yang membuat dataset ini?**
Kerangka & isi generik disusun oleh asisten AI (Claude) atas permintaan
pengembang proyek AksaraLLM. Bagian biodata pengembang individu **belum
diisi** — lihat template di bawah.

## Komposisi

`identity_core.jsonl` berisi pasangan instruksi-jawaban dalam kategori:

| Kategori | Isi | Jumlah (approx.) |
|---|---|---|
| `identity` | Nama, pembuat, tujuan, lisensi model | ~20 |
| `civics` | Pancasila, dasar negara, lambang negara | ~8 |
| `geography` | 38 provinsi + ibukota, pulau besar, gunung/laut | ~45 |
| `history` | Kemerdekaan, tokoh, peristiwa kunci | ~15 |
| `language` | Tata bahasa dasar, sapaan formal/informal | ~10 |
| `culture` | Batik, kuliner, rumah adat, tarian (contoh, bukan lengkap) | ~10 |

**Rekomendasi bobot training:** Data `identity` sebaiknya diulang beberapa
kali (mis. 20-30x) selama SFT — pola yang sama dipakai proyek-proyek LLM
lain untuk memastikan identitas "menempel" kuat meski datasetnya kecil
dibanding data instruksi umum. Kategori lain cukup 1-3x.

## Proses Pengumpulan

Fakta geografi, sejarah, dan Pancasila adalah pengetahuan publik/civic yang
stabil (bukan hasil scraping). **Catatan akurasi:** pembagian provinsi
Indonesia berubah dari waktu ke waktu (mis. pemekaran Papua tahun 2022
menjadikan totalnya 38 provinsi) — verifikasi ulang data geografi secara
berkala terhadap sumber resmi (Kemendagri) sebelum melatih model skala
besar, jangan berasumsi dokumen ini akan selalu up-to-date.

## Penggunaan yang Disarankan

- Digabung ke pipeline SFT (`aksaraLLM/sft.py`) sebagai bagian dari dataset
  instruksi yang lebih besar, bukan dataset SFT satu-satunya.
- Jangan dipakai sebagai satu-satunya sumber pengetahuan geografi/sejarah —
  ini seed data untuk *konsistensi identitas dasar*, bukan ensiklopedia.

## Distribusi & Maintenance

Berlisensi sama dengan repo `aksara-data` (Apache 2.0). Maintainer
bertanggung jawab memperbarui `identity_core.jsonl` bila ada perubahan
administratif (provinsi baru, dst) atau bila biodata pengembang di bawah
diisi/diperbarui.

---

## Template: Data Pengembang / Tim

Isi bagian ini dengan data asli, lalu (opsional) generate ulang baris
`identity` di `identity_core.jsonl` agar menyebut nama/organisasi yang benar.

```yaml
nama_proyek: AksaraLLM
organisasi: "[ISI DI SINI — mis. nama komunitas/yayasan/individu]"
pembuat_utama:
  nama: "[ISI DI SINI]"
  peran: "[ISI DI SINI — mis. Project Lead / Founder]"
  afiliasi: "[ISI DI SINI, opsional]"
  kontak: "[ISI DI SINI, opsional — email/GitHub/Discord]"
tim_inti:
  - nama: "[ISI DI SINI]"
    peran: "[ISI DI SINI]"
tanggal_mulai_proyek: "[ISI DI SINI]"
lisensi: "Apache 2.0"
kontak_umum:
  github: "https://github.com/AksaraLLM"
  discord: "https://discord.gg/aksarallm"
```
