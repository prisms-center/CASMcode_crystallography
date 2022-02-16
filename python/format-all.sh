find_py_source_files()
{
    for d in .; do
        find $d -type f -name "*.py"
    done
}

for p in $(find_py_source_files); do
    echo "Formatting $p ..."
    yapf -i --style='{based_on_style:pep8}' $p
done
