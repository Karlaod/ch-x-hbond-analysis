import os
import csv
import math
from ccdc import io


# Settings


HALOGENS = {"Cl", "Br"}

DIST_LIMITS = {
    "Cl": (2.30, 3.40),
    "Br": (2.40, 3.50),
}

ANGLE_CUTOFF = 100.0
NORMALISED_CH = 1.083

OUT_CSV = os.path.join("data", "ch_x_contacts_full.csv")

ORGANIC_ELEMENTS = {
    "H", "C", "N", "O", "F", "Cl", "Br", "I", "S", "P", "B", "Si"
}


# Geometry helpers


def distance(a, b):
    if a is None or b is None:
        return None
    if a.coordinates is None or b.coordinates is None:
        return None
    dx = a.coordinates.x - b.coordinates.x
    dy = a.coordinates.y - b.coordinates.y
    dz = a.coordinates.z - b.coordinates.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def angle_deg(a, b, c):
    if None in (a, b, c):
        return None
    if None in (a.coordinates, b.coordinates, c.coordinates):
        return None

    ba = (
        a.coordinates.x - b.coordinates.x,
        a.coordinates.y - b.coordinates.y,
        a.coordinates.z - b.coordinates.z
    )
    bc = (
        c.coordinates.x - b.coordinates.x,
        c.coordinates.y - b.coordinates.y,
        c.coordinates.z - b.coordinates.z
    )

    dot = sum(ba[i] * bc[i] for i in range(3))
    mag_ba = math.sqrt(sum(x*x for x in ba))
    mag_bc = math.sqrt(sum(x*x for x in bc))

    if mag_ba == 0 or mag_bc == 0:
        return None

    cosang = dot / (mag_ba * mag_bc)
    cosang = max(-1.0, min(1.0, cosang))

    return math.degrees(math.acos(cosang))

def normalise_hydrogen(c, h):
    if None in (c, h):
        return
    if None in (c.coordinates, h.coordinates):
        return

    dx = h.coordinates.x - c.coordinates.x
    dy = h.coordinates.y - c.coordinates.y
    dz = h.coordinates.z - c.coordinates.z

    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0:
        return

    scale = NORMALISED_CH / length

    h.coordinates = type(h.coordinates)(
        c.coordinates.x + dx * scale,
        c.coordinates.y + dy * scale,
        c.coordinates.z + dz * scale
    )


# Hybridisation classification


def classify_hybridisation(carbon):
    for bond in carbon.bonds:
        if bond.bond_type == 3:
            return "sp"
        if bond.bond_type == 2:
            return "sp2"
    return "sp3"


# Structure classification


def classify_structure(molecule):
    for atom in molecule.atoms:
        if atom.atomic_symbol not in ORGANIC_ELEMENTS:
            return "organometallic"
    return "organic"


# Main extraction


def main():

    os.makedirs("data", exist_ok=True)

    csd = io.EntryReader("CSD")

    scanned = 0
    contacts = 0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "refcode",
            "halogen",
            "h_x_distance",
            "c_h_x_angle",
            "donor_hybridisation",
            "acceptor_hybridisation",
            "donor_acceptor_pair",
            "structure_type"
        ])

        for entry in csd:
            scanned += 1

            if entry.molecule is None:
                continue

            mol = entry.molecule
            structure_type = classify_structure(mol)

            hydrogens = [a for a in mol.atoms if a.atomic_symbol == "H"]
            halogens = [a for a in mol.atoms if a.atomic_symbol in HALOGENS]

            if not hydrogens or not halogens:
                continue

            for h in hydrogens:
                carbon = next((n for n in h.neighbours if n.atomic_symbol == "C"), None)
                if carbon is None:
                    continue

                normalise_hydrogen(carbon, h)

                donor_type = classify_hybridisation(carbon)

                for x in halogens:

                    if x.coordinates is None:
                        continue

                    acceptor_c = next((n for n in x.neighbours if n.atomic_symbol == "C"), None)
                    if acceptor_c is None:
                        continue

                    acceptor_type = classify_hybridisation(acceptor_c)

                    d = distance(h, x)
                    if d is None:
                        continue

                    dmin, dmax = DIST_LIMITS[x.atomic_symbol]
                    if not (dmin <= d <= dmax):
                        continue

                    ang = angle_deg(carbon, h, x)
                    if ang is None:
                        continue

                    if ang < ANGLE_CUTOFF:
                        continue

                    pair = f"{donor_type}-{acceptor_type}"

                    writer.writerow([
                        entry.identifier,
                        x.atomic_symbol,
                        round(d, 3),
                        round(ang, 1),
                        donor_type,
                        acceptor_type,
                        pair,
                        structure_type
                    ])

                    contacts += 1

    print("\nExtraction complete")
    print(f"Structures scanned: {scanned}")
    print(f"Contacts saved: {contacts}")
    print(f"CSV written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
