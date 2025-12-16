#!/bin/bash

echo "Stylizing declarations..."
for file in $(git diff --name-only --staged | grep ".hh$"); do
    clang-format -i $file
done

echo "Stylizing definitions..."
for file in $(git diff --name-only --staged | grep ".cc$"); do
    clang-format -i $file
done

echo "Stylizing actual programs..."
for file in $(git diff --name-only --staged | grep ".cpp$"); do
    clang-format -i $file
done

touch .is_stylized.txt
