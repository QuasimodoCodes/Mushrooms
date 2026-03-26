"""
predict.py — run a single image through the trained TaxonomicYOLO26 model.

Usage
-----
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --weights docs/franken_runs/taxonomic_yolo26_5/weights/best.pt
    python predict.py path/to/image.jpg --topk 5
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Resolve project root so we can import model.py regardless of cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)

from model import TaxonomicYOLO26  # noqa: E402

# ---------------------------------------------------------------------------
# Class lists — must exactly match training order (alphabetical ImageFolder)
# ---------------------------------------------------------------------------
SPECIES = [
    "Agaricus augustus", "Agaricus xanthodermus", "Amanita amerirubescens",
    "Amanita augusta", "Amanita brunnescens", "Amanita calyptroderma",
    "Amanita citrina", "Amanita flavoconia", "Amanita muscaria",
    "Amanita pantherina", "Amanita persicina", "Amanita phalloides",
    "Amanita rubescens", "Amanita velosa", "Apioperdon pyriforme",
    "Armillaria borealis", "Armillaria mellea", "Armillaria tabescens",
    "Artomyces pyxidatus", "Bjerkandera adusta", "Bolbitius titubans",
    "Boletus edulis", "Boletus pallidus", "Boletus reticulatus",
    "Boletus rex-veris", "Calocera viscosa", "Calycina citrina",
    "Cantharellus californicus", "Cantharellus cibarius",
    "Cantharellus cinnabarinus", "Cerioporus squamosus",
    "Cetraria islandica", "Chlorociboria aeruginascens",
    "Chlorophyllum brunneum", "Chlorophyllum molybdites",
    "Chondrostereum purpureum", "Cladonia fimbriata",
    "Cladonia rangiferina", "Cladonia stellaris", "Clitocybe nebularis",
    "Clitocybe nuda", "Coltricia perennis", "Coprinellus disseminatus",
    "Coprinellus micaceus", "Coprinopsis atramentaria",
    "Coprinopsis lagopus", "Coprinus comatus", "Crucibulum laeve",
    "Cryptoporus volvatus", "Daedaleopsis confragosa",
    "Daedaleopsis tricolor", "Entoloma abortivum", "Evernia mesomorpha",
    "Evernia prunastri", "Flammulina velutipes", "Fomes fomentarius",
    "Fomitopsis betulina", "Fomitopsis mounceae", "Fomitopsis pinicola",
    "Galerina marginata", "Ganoderma applanatum", "Ganoderma curtisii",
    "Ganoderma oregonense", "Ganoderma tsugae", "Gliophorus psittacinus",
    "Gloeophyllum sepiarium", "Graphis scripta", "Grifola frondosa",
    "Gymnopilus luteofolius", "Gyromitra esculenta", "Gyromitra gigas",
    "Gyromitra infula", "Hericium coralloides", "Hericium erinaceus",
    "Hygrophoropsis aurantiaca", "Hypholoma fasciculare",
    "Hypholoma lateritium", "Hypogymnia physodes",
    "Hypomyces lactifluorum", "Imleria badia", "Inonotus obliquus",
    "Ischnoderma resinosum", "Kuehneromyces mutabilis",
    "Laccaria ochropurpurea", "Lactarius deliciosus",
    "Lactarius torminosus", "Lactarius turpis", "Laetiporus sulphureus",
    "Leccinum albostipitatum", "Leccinum aurantiacum", "Leccinum scabrum",
    "Leccinum versipelle", "Lepista nuda", "Leratiomyces ceres",
    "Leucoagaricus americanus", "Leucoagaricus leucothites",
    "Lobaria pulmonaria", "Lycogala epidendrum", "Lycoperdon perlatum",
    "Lycoperdon pyriforme", "Macrolepiota procera", "Merulius tremellosus",
    "Mutinus ravenelii", "Mycena haematopus", "Mycena leaiana",
    "Nectria cinnabarina", "Omphalotus illudens", "Omphalotus olivascens",
    "Panaeolus papilionaceus", "Panellus stipticus", "Parmelia sulcata",
    "Paxillus involutus", "Peltigera aphthosa", "Peltigera praetextata",
    "Phaeolus schweinitzii", "Phaeophyscia orbicularis",
    "Phallus impudicus", "Phellinus igniarius", "Phellinus tremulae",
    "Phlebia radiata", "Phlebia tremellosa", "Pholiota aurivella",
    "Pholiota squarrosa", "Phyllotopsis nidulans", "Physcia adscendens",
    "Platismatia glauca", "Pleurotus ostreatus", "Pleurotus pulmonarius",
    "Psathyrella candolleana", "Pseudevernia furfuracea",
    "Pseudohydnum gelatinosum", "Psilocybe azurescens",
    "Psilocybe caerulescens", "Psilocybe cubensis",
    "Psilocybe cyanescens", "Psilocybe ovoideocystidiata",
    "Psilocybe pelliculosa", "Retiboletus ornatipes",
    "Rhytisma acerinum", "Sarcomyxa serotina", "Sarcoscypha austriaca",
    "Sarcosoma globosum", "Schizophyllum commune", "Stereum hirsutum",
    "Stereum ostrea", "Stropharia aeruginosa", "Stropharia ambigua",
    "Suillus americanus", "Suillus granulatus", "Suillus grevillei",
    "Suillus luteus", "Suillus spraguei", "Tapinella atrotomentosa",
    "Trametes betulina", "Trametes gibbosa", "Trametes hirsuta",
    "Trametes ochracea", "Trametes versicolor", "Tremella mesenterica",
    "Trichaptum biforme", "Tricholoma murrillianum",
    "Tricholomopsis rutilans", "Tylopilus felleus",
    "Tylopilus rubrobrunneus", "Urnula craterium", "Verpa bohemica",
    "Volvopluteus gloiocephalus", "Vulpicida pinastri",
    "Xanthoria parietina",
]

GENERA = sorted(set(s.split()[0] for s in SPECIES))

# ---------------------------------------------------------------------------
# Image pre-processing (must match eval transform from dataset.py)
# ---------------------------------------------------------------------------
_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def predict(image_path: str, weights_path: str, topk: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading weights from: {weights_path}")
    model = TaxonomicYOLO26(
        num_genera=len(GENERA),
        num_species=len(SPECIES),
        weights=os.path.join(_ROOT, "yolo26n-cls.pt"),
    )
    ckpt = torch.load(weights_path, map_location=device)
    # Support both raw state_dict and checkpoint dict formats
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        genus_logits, species_logits = model(tensor)

    genus_probs   = F.softmax(genus_logits,   dim=1)[0]
    species_probs = F.softmax(species_logits, dim=1)[0]

    # Top-k results
    topk = min(topk, len(SPECIES))
    sp_topk   = species_probs.topk(topk)
    gen_topk  = genus_probs.topk(min(topk, len(GENERA)))

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Device: {device}\n")

    print(f"{'SPECIES':=<45}")
    for prob, idx in zip(sp_topk.values, sp_topk.indices):
        bar = "#" * int(prob.item() * 40)
        print(f"  {SPECIES[idx]:<38}  {prob.item()*100:5.1f}%  {bar}")

    print(f"\n{'GENUS':=<45}")
    for prob, idx in zip(gen_topk.values, gen_topk.indices):
        bar = "#" * int(prob.item() * 40)
        print(f"  {GENERA[idx]:<38}  {prob.item()*100:5.1f}%  {bar}")

    # Consistency check
    top_species = SPECIES[sp_topk.indices[0].item()]
    top_genus   = GENERA[gen_topk.indices[0].item()]
    expected_genus = top_species.split()[0]
    if top_genus != expected_genus:
        print(f"\n  [!] Heads disagree — species implies '{expected_genus}' but genus head picked '{top_genus}'")
    else:
        print(f"\n  [OK] Both heads agree on genus '{top_genus}'")


def main():
    parser = argparse.ArgumentParser(description="Predict mushroom species from an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--weights",
        default=os.path.join(_ROOT, "docs", "franken_runs", "taxonomic_yolo26_5", "weights", "best.pt"),
        help="Path to model weights (.pt)",
    )
    parser.add_argument("--topk", type=int, default=3, help="How many top predictions to show")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: image not found: {args.image}")
        sys.exit(1)
    if not os.path.exists(args.weights):
        print(f"ERROR: weights not found: {args.weights}")
        sys.exit(1)

    predict(args.image, args.weights, args.topk)


if __name__ == "__main__":
    main()
