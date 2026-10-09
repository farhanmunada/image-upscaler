# AGENTS.md

> Aturan universal agen coding. Salin file ini ke root setiap projek baru.
> Knowledge base lintas-projek: `C:\Users\vola\agentVault`

---

## 1. Aturan Inti

1. **Keamanan deny-first** — jangan baca/tulis/cetak/commit file rahasia (`.env`, `.env.*`, `*.pem`, `id_rsa`, token cloud).
2. **Jangan curangi tes** — perbaiki kode bisnis, bukan asersi tes. Jangan hapus/lemahkan tes agar hijau.
3. **Jangan halusinasi dependensi** — urutan: stdlib → paket terpasang → fungsi mandiri → minta izin manusia sebelum menambah paket.
4. **Batas 3 kali** — gagal 3x berturut di target sama: stop, rollback via git, lapor hipotesis + error ke manusia.
5. **Jangan klaim selesai palsu** — linter, build, typecheck, dan tes harus hijau sebelum bilang selesai.
6. **Tulis ke file, bukan chat** — riset, PRD, arsitektur ditulis ke `docs/`. Chat hanya status ringkas (maks 5 baris).
7. **Jangan mengarang fakta** — setiap klaim faktual harus punya sumber (file dibaca / output perintah / docs resmi). Belum yakin → nyatakan "belum diverifikasi".

---

## 2. Alur Kerja

1. **Intake** — baca `AGENTS.md` + seluruh `docs/`.
2. **Riset** — eksplor codebase & docs resmi → tulis ke `docs/RESEARCH.md`.
3. **Spec** — perjelas intent → tulis PRD ke `docs/PRD.md` (Bahasa Indonesia).
4. **[GERBANG 1]** — berhenti, tunggu approval manusia.
5. **Arsitektur** — pecah tugas + rancangan → `docs/ARCHITECTURE.md`.
6. **[GERBANG 2]** — berhenti bila tugas ≥3 file / ubah skema DB, tunggu approval.
7. **Implementasi** — kode bersih, incremental, verifikasi tiap langkah.
8. **Verifikasi** — typecheck + lint + unit test, lalu smoke test nyata (DB/migrasi).
9. **Review** — regresi, keamanan, diff bersih.
10. **Commit** — commit atomik konvensional (`feat(...)`, `fix(...)`).

---

## 3. Gerbang Manusia (STOP & WAIT)

- **Gerbang 1 — PRD:** setelah `docs/PRD.md` selesai, berhenti. Jangan tulis kode/skema sebelum disetujui.
- **Gerbang 2 — Arsitektur:** untuk tugas ≥3 file atau perubahan skema DB, paparkan rencana lalu berhenti sebelum file kode pertama.

---

## 4. Bahasa

- Semua dokumen di `docs/` → **Bahasa Indonesia** (istilah teknis/identifier tetap Inggris).
- Kode & komentar → **Bahasa Inggris**.

---

## 5. Skills

Skill di-install **global** (`~/.agents/skills/<nama>/SKILL.md`) dan aktif otomatis di semua projek. Router utama: **`using-agent-skills`**.

**Aturan Eksekusi Skill:**
1. Jika tugas cocok dengan skill (bahkan kemungkinan kecil), **panggil skill dulu** sebelum bertindak.
2. Ikuti workflow skill secara ketat — jangan terapkan sebagian.
3. Jangan lewati langkah wajib (spec, plan, test) yang diminta skill.

**Pemetaan Otomatis Intent → Skill:**

| Intent | Skill |
|---|---|
| Fitur / fungsionalitas baru | `spec-driven-development` → `incremental-implementation` → `test-driven-development` |
| Perencanaan / breakdown | `planning-and-task-breakdown` |
| Bug / error / perilaku tak terduga | `debugging-and-error-recovery` |
| Code review | `code-review-and-quality` |
| Refactoring / simplifikasi | `code-simplification` |
| Desain API / interface | `api-and-interface-design` |
| UI / styling / design system | `ui-ux-pro-max` → `frontend-ui-engineering` |
| Brand / logo / visual assets | `design` / `brand` |
| Ide mentah / brainstorm | `idea-refine` → `interview-me` |
| Tetapkan standar mutu | `constraint-driven-development` |
| Konfigurasi konteks / rules | `context-engineering` |
| Verifikasi terhadap docs resmi | `source-driven-development` |
| Verifikasi berlapis / adversarial | `doubt-driven-development` |
| Test di browser nyata | `browser-testing-with-devtools` |
| Keamanan / audit OWASP | `security-and-hardening` |
| Performa / query lambat | `performance-optimization` |
| Logging / metrik / tracing | `observability-and-instrumentation` |
| Git / commit / release | `git-workflow-and-versioning` |
| Dokumentasi / ADR | `documentation-and-adrs` |
| CI/CD pipeline | `ci-cd-and-automation` |
| Launch / deploy produksi | `shipping-and-launch` |
| Migrasi / deprecation | `deprecation-and-migration` |
| Riset web / baca docs versi baru | `agent-browser` |

---

## 6. Knowledge Base (baca sesuai kebutuhan)

Baca node vault **hanya bila butuh detail** — jangan muat semuanya di awal sesi.

| Topik | Lokasi |
|---|---|
| Inisiasi & PRD | `agentVault\01-Initiation-and-Spec\` |
| Arsitektur & tech stack | `agentVault\02-Architecture-and-Tech\` |
| Siklus hidup kerja | `agentVault\02-Execution-Lifecycle\` |
| Clean code | `agentVault\03-Implementation-and-Clean-Code\` |
| Testing | `agentVault\04-Quality-and-Repeated-Testing\` |
| Keamanan | `agentVault\04-Security\` |
| Anti-pattern & recovery | `agentVault\03-Anti-Patterns\` |
| Aturan inti | `agentVault\01-Core-Rules\` |

Indeks lengkap: `agentVault\INDEX.md`

---

## 7. Matriks Perizinan

| Risiko | Aksi | Kebijakan |
|---|---|---|
| Rendah | read, grep, glob, git status/diff, unit test | Auto-allow |
| Sedang | edit kode, tambah dependensi, migrasi DB | Konfirmasi |
| Tinggi | `rm -rf`, `git push -f`, edit `.env`, drop DB | Blokir / tanya manusia |

---

## 8. Perintah Verifikasi

> Sesuaikan dengan stack projek:

- **Lint:** `npm run lint` / `ruff check .`
- **Typecheck:** `npx tsc --noEmit` / `mypy .`
- **Test:** `npm test` / `pytest`
- **Build:** `npm run build`
