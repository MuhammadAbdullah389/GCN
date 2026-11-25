from androguard.misc import AnalyzeAPK

apk_path = r"data\raw\benign\app-debug.apk"

apk, dex_files, analysis = AnalyzeAPK(apk_path)

for dex in dex_files:
    found = False

    for class_obj in dex.get_classes():
        for method in class_obj.get_methods():

            if method.get_code() is None:
                continue

            for instruction in method.get_instructions():

                if instruction.get_name().startswith("invoke-"):

                    print("Instruction:", instruction.get_name())
                    print("Output:", instruction.get_output())
                    print("Operands:", instruction.get_operands())

                    found = True
                    break

            if found:
                break

        if found:
            break

    if found:
        break