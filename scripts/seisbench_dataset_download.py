import seisbench.data as sbd
import seisbench

seisbench.cache_root = "/data/wsd04/ak287/.seisbench"

# ── Already downloaded — skip unless you want to force refresh ────────────
"""
print("Downloading stead...")
sbd.STEAD()
print("Downloading instancecounts...")
sbd.InstanceCounts()
print("Downloading pnw...")
sbd.PNW()
print("Downloading txed...")
sbd.TXED()
print("Downloading mlaapde...")
sbd.MLAAPDE()
print("Downloading iquique...")
sbd.Iquique()
print("Downloading scedc...")
sbd.SCEDC()


# ── New datasets recommended by Münchmeyer ────────────────────────────────


print("Downloading ETHZ (~0.5 GB — Switzerland, high-quality labels)...")
sbd.ETHZ(wait_for_file=True)



print("Downloading VCSEIS (~0.1 GB — volcanic data)...")
sbd.VCSEIS(force =True)


print("Downloading PiSDL (induced seismicity near-field)...")
sbd.PiSDL(force =True)






print("Downloading CREW (good data quality, broadband)...")
sbd.CREW(force =True)

# ── Datasets that may not have SeisBench classes yet ──────────────────────
# Try these — they may fail if not in your SeisBench version
for name, cls_name in [("AQ2009", "AQ2009"), ("CWA", "CWA")]:
    try:
        cls = getattr(sbd, cls_name)
        print(f"Downloading {name}...")
        cls()
    except AttributeError:
        print(f"SKIP {name}: not available as sbd.{cls_name} in your SeisBench version")
        print(f"  → Check: https://seisbench.readthedocs.io for the correct class name")
    except Exception as e:
        print(f"FAILED {name}: {e}")
        
  

print("Downloading GEOFON (for teleseismic P — P-only evaluation)...")
# Note: already in cache but may only have metadata; this ensures waveforms
sbd.GEOFON(force=True)

"""

print("Downloading CEED (~0.5 GB — replaces SCEDC as California dataset)...")
sbd.CEED(force=True)




print("\nAll done.")