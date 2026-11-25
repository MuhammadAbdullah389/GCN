
import sys
import json
from pathlib import Path

from androguard.misc import AnalyzeAPK


def normalize_descriptor(descriptor):
    """Remove unnecessary whitespace from a method descriptor."""
    return "".join(descriptor.split())


def get_method_key(method):
    """Create a unique identifier for a method."""
    return (
        method.get_class_name(),
        method.get_name(),
        normalize_descriptor(method.get_descriptor())
    )


def is_application_class(class_name, package_path):
    """Check whether a class belongs to the application package."""
    return (
        class_name.startswith(package_path + "/")
        or class_name == package_path + ";"
    )


def is_resource_class(class_name):
    """Exclude generated Android resource classes."""
    simple_name = class_name.rstrip(";").split("/")[-1]

    return (
        simple_name == "R"
        or simple_name.startswith("R$")
    )


def extract_api_calls(method, package_path):
    """Extract complete external method references called by a method."""
    api_calls = set()

    if method.get_code() is None:
        return []

    for instruction in method.get_instructions():

        if not instruction.get_name().startswith("invoke-"):
            continue

        operands = instruction.get_operands()

        for operand in operands:

            if len(operand) < 3:
                continue

            method_reference = operand[-1]

            if not isinstance(method_reference, str):
                continue

            if "->" not in method_reference:
                continue

            # Remove display-format whitespace
            method_reference = "".join(method_reference.split())

            target_class = method_reference.split("->")[0]

            # Exclude calls to application-defined methods
            if not is_application_class(target_class, package_path):
                api_calls.add(method_reference)

    return sorted(api_calls)


def parse_apk(apk_path):
    apk_path = Path(apk_path).resolve()

    if not apk_path.exists():
        raise FileNotFoundError(f"APK not found: {apk_path}")

    if apk_path.suffix.lower() != ".apk":
        raise ValueError("The supplied file must have an .apk extension.")

    print(f"\nAnalyzing APK: {apk_path.name}")

    # Androguard logs remain visible in the terminal
    apk, dex_files, analysis = AnalyzeAPK(str(apk_path))

    package_name = apk.get_package()
    package_path = "L" + package_name.replace(".", "/")

    print(f"\nPackage: {package_name}")

    # --------------------------------------------------
    # STEP 1: Extract application-defined methods
    # --------------------------------------------------

    application_methods = []

    for dex in dex_files:
        for class_obj in dex.get_classes():

            class_name = class_obj.get_name()

            if not is_application_class(class_name, package_path):
                continue

            if is_resource_class(class_name):
                continue

            for method in class_obj.get_methods():

                # Ignore abstract/native methods without executable code
                if method.get_code() is None:
                    continue

                application_methods.append(method)

    # Assign a unique integer ID to every graph node
    method_ids = {}

    for node_id, method in enumerate(application_methods):
        method_ids[get_method_key(method)] = node_id

    print(f"Application methods selected: {len(application_methods)}")

    # --------------------------------------------------
    # STEP 2: Extract API calls for every method
    # --------------------------------------------------

    nodes = []

    for method in application_methods:

        method_key = get_method_key(method)
        node_id = method_ids[method_key]

        api_calls = extract_api_calls(method, package_path)

        nodes.append({
            "id": node_id,
            "class": method.get_class_name(),
            "name": method.get_name(),
            "descriptor": normalize_descriptor(
                method.get_descriptor()
            ),
            "api_calls": api_calls
        })

    print("API calls extracted.")

    # --------------------------------------------------
    # STEP 3: Extract method-to-method relationships
    # --------------------------------------------------

    edges = set()

    for method_analysis in analysis.get_methods():

        source_method = method_analysis.get_method()
        source_key = get_method_key(source_method)

        # Only application methods can be source nodes
        if source_key not in method_ids:
            continue

        source_id = method_ids[source_key]

        for _, target_analysis, _ in method_analysis.get_xref_to():

            target_method = target_analysis.get_method()
            target_key = get_method_key(target_method)

            # Keep only calls to other application methods
            if target_key not in method_ids:
                continue

            target_id = method_ids[target_key]

            edges.add((source_id, target_id))

    edge_list = [
        {
            "source": source,
            "target": target
        }
        for source, target in sorted(edges)
    ]

    print(f"Method relationships extracted: {len(edge_list)}")

    # --------------------------------------------------
    # STEP 4: Prepare processed JSON
    # --------------------------------------------------

    processed_data = {
        "apk_name": apk_path.name,
        "package_name": package_name,
        "node_count": len(nodes),
        "edge_count": len(edge_list),
        "methods": nodes,
        "edges": edge_list
    }

    # Save relative to the project root
    project_root = Path(__file__).resolve().parents[2]

    output_dir = project_root / "data" / "processed" / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{apk_path.stem}.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(processed_data, file, indent=4)

    print("\n========== EXTRACTION SUMMARY ==========")
    print(f"APK: {apk_path.name}")
    print(f"Application methods: {len(nodes)}")
    print(f"Graph edges: {len(edge_list)}")
    print(f"JSON saved to: {output_file}")
    print("=========================================")

    return processed_data


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            'Usage: python src/extraction/apk_parser.py "path/to/file.apk"'
        )
        sys.exit(1)

    parse_apk(sys.argv[1])
