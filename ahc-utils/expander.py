if __name__ == "__main__":
    main_file = "src/main.rs"

    src = []

    with open(main_file, "r") as f:
        for line in f:
            if len(line) >= 3 and line[:3] == "mod":
                lib_name = line.split(" ")[1][:-2]
                src.append(f"pub mod {lib_name} {{\n")
                with open(f"src/{lib_name}.rs", "r") as f:
                    is_test = False
                    for line in f:
                        if len(line) >= 7 and line[:7] == "#[test]":
                            is_test = True
                        elif is_test:
                            if len(line) >= 1 and line[:1] == "}":
                                is_test = False
                        else:
                            src.append("    " + line)
                src.append("}\n")
            else:
                src.append(line)

    for line in src:
        print(line.rstrip())
