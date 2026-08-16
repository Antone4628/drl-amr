// conv_family_periodic.geo — convergence-study mesh family for AdvDiff advection (DG Phase 5).
// Replicates hexa_TFI_10x20_periodic.msh: domain [-5,5]x[0,20], doubly periodic, transfinite quads.
// Tag convention (measured, DG roadmap §5.15): the tag names the axis the edge RUNS ALONG —
//   periodicx = bottom+top (edges along x); periodicz = left+right (edges along y).
//   The builder's per-element axis vote makes the name's axis claim non-load-bearing, but the
//   reader must recognize both names as periodic. Verify against the existing 10x20 mesh before use.
// Generate (from anywhere):
//   gmsh conv_family_periodic.geo -2 -setnumber nelemx <NX> -setnumber nelemy <NY> -o <out.msh>

If (!Exists(nelemx))
  nelemx = 10;
EndIf
If (!Exists(nelemy))
  nelemy = 20;
EndIf

xmin = -5; xmax = 5;
ymin =  0; ymax = 20;
gridsize = (xmax - xmin) / nelemx;

Point(1) = {xmin, ymin, gridsize};
Point(2) = {xmax, ymin, gridsize};
Point(3) = {xmax, ymax, gridsize};
Point(4) = {xmin, ymax, gridsize};

Line(1) = {1, 2};  // bottom (runs along x)
Line(2) = {2, 3};  // right  (runs along y)
Line(3) = {3, 4};  // top    (runs along x)
Line(4) = {4, 1};  // left   (runs along y)

npx = nelemx + 1;
npy = nelemy + 1;

Transfinite Line {1, 3} = npx;
Transfinite Line {4, -2} = npy Using Progression 1.0;

Line Loop(11) = {4, 1, 2, 3};
Plane Surface(12) = {11};
Transfinite Surface {12};
Recombine Surface {12};

Physical Point("boundary", 1) = {1, 2, 3, 4};
Physical Curve("periodicx", 2) = {1, 3};
Physical Curve("periodicz", 3) = {2, 4};
Physical Surface("domain", 4) = {12};