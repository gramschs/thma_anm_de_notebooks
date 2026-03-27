#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='This script converts as Myst markdown file to a Jupyter notebook.')
    parser.add_argument('filename', type=str)

    # no arguments given
    if len(sys.argv)==1:
        print('Provide a filename: convert_myst_to_notebook.py inputfile.md')
        sys.exit(1)

    return parser.parse_args()


def clean_myst_file(input_filename, output_filename):
    counter_miniexercise = 1

    output_lines = []
    with open(input_filename) as file:
        inside_learning_goals = False
        inside_miniexercise_with_three_ticks = False
        inside_exercise_with_three_ticks = False
        inside_exercise_with_four_ticks = False
        inside_solution_with_three_ticks = False
        inside_solution_with_four_ticks = False

        for line in file:

            # Deleting Learning Goals
            if '```{admonition} Lernziele' in line:
                inside_learning_goals= True
                continue

            if 'class: attention' in line and inside_learning_goals:
                continue

            if '```' in line and inside_learning_goals:
                inside_learning_goals = False
                continue

            # Deleting Mini Exercise, i.e. ```{admonition} Mini-Übung (3 ticks)
            if '```{admonition} Mini-Übung' in line:
                # print('### Mini-Übung 3 Ticks Start')
                inside_miniexercise_with_three_ticks = True
                output_lines.append(f'### Mini-Übung {counter_miniexercise}\n\n')
                counter_miniexercise += 1 
                continue

            if ':class: tip' in line and inside_miniexercise_with_three_ticks:
                continue

            if '```' in line and inside_miniexercise_with_three_ticks:
                # print('### Mini-Übung 3 Ticks Ende')
                inside_miniexercise_with_three_ticks = False
                continue

            # Deleting ````{admonition} Lösung (four ticks)
            if '````{admonition} Lösung' in line:
                inside_solution_with_four_ticks = True
                #print('### Lösung 4 Ticks Start')
                continue

            if inside_solution_with_four_ticks:
                if '````' in line:
                    inside_solution_with_four_ticks = False
                    #print('### Lösung 4 Ticks Ende')
                    continue
                else:
                    continue
                
            # Deleting ```{admonition} Lösung (three ticks)
            if '```{admonition} Lösung' in line:
                #print('### Lösung 3 Ticks Start')
                inside_solution_with_three_ticks = True
                continue

            if inside_solution_with_three_ticks:
                if '```' in line:
                    #print('### Lösung 3 Ticks Ende')
                    inside_solution_with_three_ticks = False
                    continue
                else:
                    continue

            # Deleting Exercise, i.e. ````{admonition} Übung (four ticks)
            if '````{admonition} Übung' in line:
                #print('### Übung 4 Ticks Start')
                parts = line.split('Übung')
                inside_exercise_with_four_ticks = True
                output_lines.append(f'## Übung {parts[-1].strip()}\n\n')
                continue

            if ':class: tip' in line and inside_exercise_with_four_ticks:
                continue

            if '````' in line and inside_exercise_with_four_ticks:
                #print('### Übung 4 Ticks Ende')
                inside_exercise_with_four_ticks = False
                continue

            # Deleting Exercise, i.e. ```{admonition} Übung (three ticks)
            if '```{admonition} Übung' in line:
                #print('### Übung Start')
                parts = line.split('Übung')
                inside_exercise_with_three_ticks = True
                output_lines.append(f'## Übung {parts[-1].strip()}\n\n')
                continue

            if ':class: tip' in line and inside_exercise_with_three_ticks:
                continue

            if '```' in line and inside_exercise_with_three_ticks:
                #print('### Übung Ende')
                inside_exercise_with_three_ticks = False
                continue


            else:
                output_lines.append(line)

    # remove double blank line
    cleaned_lines = []
    prev_blank = False
    for line in output_lines:
        is_blank = line.strip() == ''
        if is_blank and prev_blank:
            continue
        cleaned_lines.append(line)
        prev_blank = is_blank
    output_lines = cleaned_lines

    warning_tags = [':class:', '```{figure}', '```{mermaid}', '```{dropdown}']
    for line_number, line in enumerate(output_lines):
        for tag in warning_tags:
            if tag in line:
                print(f'warning line no {line_number}: {line[:-1]}')

    with open(output_filename, 'w') as output_file:
        for line in output_lines:
            output_file.write(line)
    return

def main():

    args = parse_command_line_arguments()
    input_filename = args.filename

    output_filename = Path(".") / Path(args.filename).name
    clean_myst_file(input_filename, output_filename)


if __name__ == "__main__":
    main()


from pathlib import Path
