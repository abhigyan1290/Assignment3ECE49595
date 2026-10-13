| model                        | task               | mode   | ensemble   |   n_queries |   P@1 |   P@3 |   P@5 |   R@10 |   MRR |   mAP |   chance |
|:-----------------------------|:-------------------|:-------|:-----------|------------:|------:|------:|------:|-------:|------:|------:|---------:|
| openai/clip-vit-base-patch32 | item name          | full   | False      |           5 | 1.000 | 1.000 | 1.000 |  0.867 | 1.000 | 0.927 |    0.168 |
| openai/clip-vit-base-patch32 | item name          | full   | True       |           5 | 1.000 | 0.867 | 0.880 |  0.758 | 1.000 | 0.791 |    0.168 |
| openai/clip-vit-base-patch32 | lost report        | full   | False      |          15 | 1.000 | 0.933 | 0.880 |  0.742 | 1.000 | 0.811 |    0.168 |
| openai/clip-vit-base-patch32 | item name          | grid   | True       |           5 | 0.800 | 0.800 | 0.800 |  0.715 | 0.829 | 0.754 |    0.168 |
| openai/clip-vit-base-patch32 | lost report        | grid   | False      |          15 | 0.933 | 0.911 | 0.867 |  0.734 | 0.967 | 0.785 |    0.168 |
| openai/clip-vit-base-patch32 | item name          | box    | True       |           5 | 1.000 | 1.000 | 1.000 |  0.910 | 1.000 | 0.924 |    0.168 |
| openai/clip-vit-base-patch32 | lost report        | box    | False      |          15 | 0.933 | 0.956 | 0.907 |  0.812 | 0.967 | 0.860 |    0.168 |
| openai/clip-vit-base-patch32 | image->image       | full   | False      |           5 | 1.000 | 1.000 | 0.920 |  0.752 | 1.000 | 0.864 |    0.184 |
| openai/clip-vit-base-patch32 | text+image (a=0.5) | full   | False      |           5 | 1.000 | 1.000 | 0.960 |  0.809 | 1.000 | 0.898 |    0.184 |
| openai/clip-vit-base-patch32 | image->image       | grid   | False      |           5 | 1.000 | 1.000 | 1.000 |  0.820 | 1.000 | 0.914 |    0.184 |
| openai/clip-vit-base-patch32 | text+image (a=0.5) | grid   | False      |           5 | 1.000 | 1.000 | 1.000 |  0.834 | 1.000 | 0.925 |    0.184 |
| openai/clip-vit-base-patch16 | item name          | full   | False      |           5 | 1.000 | 1.000 | 0.960 |  0.817 | 1.000 | 0.878 |    0.168 |
| openai/clip-vit-base-patch16 | item name          | full   | True       |           5 | 0.800 | 0.800 | 0.760 |  0.715 | 0.820 | 0.749 |    0.168 |
| openai/clip-vit-base-patch16 | lost report        | full   | False      |          15 | 0.933 | 0.911 | 0.867 |  0.710 | 0.967 | 0.786 |    0.168 |
| openai/clip-vit-base-patch16 | item name          | grid   | True       |           5 | 0.800 | 0.800 | 0.800 |  0.715 | 0.850 | 0.756 |    0.168 |
| openai/clip-vit-base-patch16 | lost report        | grid   | False      |          15 | 0.867 | 0.844 | 0.813 |  0.748 | 0.913 | 0.768 |    0.168 |
| openai/clip-vit-base-patch16 | item name          | box    | True       |           5 | 1.000 | 1.000 | 1.000 |  0.860 | 1.000 | 0.914 |    0.168 |
| openai/clip-vit-base-patch16 | lost report        | box    | False      |          15 | 1.000 | 0.978 | 0.947 |  0.849 | 1.000 | 0.874 |    0.168 |
| openai/clip-vit-base-patch16 | image->image       | full   | False      |           5 | 1.000 | 1.000 | 1.000 |  0.859 | 1.000 | 0.926 |    0.184 |
| openai/clip-vit-base-patch16 | text+image (a=0.5) | full   | False      |           5 | 1.000 | 1.000 | 1.000 |  0.816 | 1.000 | 0.912 |    0.184 |
| openai/clip-vit-base-patch16 | image->image       | grid   | False      |           5 | 1.000 | 1.000 | 1.000 |  0.895 | 1.000 | 0.958 |    0.184 |
| openai/clip-vit-base-patch16 | text+image (a=0.5) | grid   | False      |           5 | 1.000 | 1.000 | 1.000 |  0.852 | 1.000 | 0.938 |    0.184 |