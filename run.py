"""Script de lancement du backend EduStat TN"""
import subprocess, sys, time, urllib.request, json, os

os.chdir("C:/Users/dali/Desktop/edustat2")

print("Demarrage du backend FastAPI...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.main:app", "--port", "8000", "--reload"],
    creationflags=subprocess.CREATE_NEW_CONSOLE,
    cwd="C:/Users/dali/Desktop/edustat2"
)

print("Attente du demarrage (max 15s)...")
for i in range(150):
    time.sleep(0.1)
    try:
        r = urllib.request.urlopen("http://127.0.0.1:8000/", timeout=1)
        d = json.loads(r.read())
        print("\n=== BACKEND UP ===")
        print("  Model    : %s" % d["model"])
        print("  Accuracy : %.2f%%" % (d["accuracy"] * 100))
        print("  F1       : %.2f%%" % (d["f1_weighted"] * 100))
        break
    except Exception:
        if i % 20 == 0 and i > 0:
            print("  ... toujours en attente (%ds)" % (i // 10))
else:
    print("ERREUR: Le backend n a pas demarré après 15s.")
    proc.terminate()
    sys.exit(1)

print("\n=== TEST DES ENDPOINTS ===")

# /api/sections
try:
    r = urllib.request.urlopen("http://127.0.0.1:8000/api/sections", timeout=5)
    sections = json.loads(r.read())
    print("  OK /api/sections — %d sections: %s" % (
        len(sections), ", ".join(s["code"] for s in sections)))
except Exception as e:
    print("  FAIL /api/sections:", e)

# /api/stats
try:
    r = urllib.request.urlopen("http://127.0.0.1:8000/api/stats", timeout=5)
    stats = json.loads(r.read())
    print("  OK /api/stats — %d filieres, %d universites, score_moyen=%.2f" % (
        stats["total_filieres"], stats["total_universites"], stats["score_moyen_2025"]))
except Exception as e:
    print("  FAIL /api/stats:", e)

# /api/recommend
try:
    body = json.dumps({"section_bac": "M", "score_etudiant": 150.0, "top_n": 3}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/api/recommend",
          data=body, headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=5)
    rec = json.loads(r.read())
    print("  OK /api/recommend — %d filieres éligibles" % rec["total_eligible"])
    for i, x in enumerate(rec["recommendations"][:3], 1):
        print("     %d. %s | seuil=%.2f | marge=%.2f" % (
            i, x["Filiere"], x["seuil_admission"], x["marge"]))
except Exception as e:
    print("  FAIL /api/recommend:", e)

# /api/predict
try:
    body = json.dumps({"section_bac": "S", "score_2025": 120.0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/api/predict",
          data=body, headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=5)
    pred = json.loads(r.read())
    print("  OK /api/predict — domaine=%s (%.1f%%)" % (
        pred["domaine_predit"], pred["confiance_pct"]))
    for t in pred["top_5_domaines"][:3]:
        print("     - %s : %.1f%%" % (t["domaine"], t["probabilite_pct"]))
except Exception as e:
    print("  FAIL /api/predict:", e)

print("\n=== LIENS ===")
print("  API backend  : http://127.0.0.1:8000")
print("  Swagger UI   : http://127.0.0.1:8000/docs")
print("  Frontend     : ouvrir C:/Users/dali/Desktop/edustat2/frontend/index.html")
print("\nServeur en cours... (Ctrl+C pour arreter)")
proc.wait()
