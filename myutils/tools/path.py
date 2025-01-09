# -*- coding: utf-8 -*-
"""path-relative operations.
This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.
Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::
        $ python example_google.py
Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.
Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.
        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.
Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension
.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html
"""






import os
import shutil



def clear_folder(folder_path: str):
    """clear all the files, links, or sub-folders for a given folder

    Args:
        folder_path: the absolute or relative path of the folder
    
    Returns:
        None.

    Examples:
        to use the function.
        >>> path = "./tmp_folder"
        >>> clear_folder(path)
        f'The folder {folder_path} does not exist or is not a directory.'
    """
    
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Remove all contents of the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The folder {folder_path} does not exist or is not a directory.')