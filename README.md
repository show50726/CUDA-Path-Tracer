CUDA Path Tracer
================

Scene files define reusable geometry in `Geometries` and place it in the scene
through `Instances`. Geometry IDs are unique nonnegative integers. Each instance
references a geometry with `GEOMETRY` and supplies its own material and transform:

```json
"Geometries": [
    { "ID": 0, "TYPE": "sphere" }
],
"Instances": [
    {
        "GEOMETRY": 0,
        "MATERIAL": "light",
        "TRANS": [0, 5, 0],
        "ROTAT": [0, 0, 0],
        "SCALE": [3, 3, 3]
    }
]
```

Supported geometry types are `sphere`, `cube`, `triangle` (with local-space
`V0`, `V1`, `V2`), and `mesh` (with an OBJ `PATH`, relative to the working
directory). Multiple instances share the same loaded triangle or mesh data.
Sphere and cube use the existing unit primitives; adjust their size with the
instance's `SCALE`. Instance `ROTAT` is in degrees. The former `Objects`,
instance `TYPE`, and `MESHID` fields are replaced by this format.

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

