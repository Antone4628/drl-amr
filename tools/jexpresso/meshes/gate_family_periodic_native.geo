// gate_family_periodic_native.geo — the conv_family_periodic.geo family with NATIVE gmsh periodicity.
// Same domain [-5,5]x[0,20], same transfinite quads, same physical tags as conv_family_periodic.geo.
// Two differences, both deliberate:
//   1. Points sit at z = 0 (mesh size moved to the fourth slot). conv_family_periodic.geo writes the
//      mesh size in the z slot, which makes GridapGmsh report Dp = 3; Jexpresso's flatten helper then
//      rebuilds the topology from the grid's node ids and discards any periodic glue.
//   2. Two Periodic Curve constraints, so the .msh carries a $Periodic section and GridapGmsh glues
//      the paired vertices in the model topology (geometry stays per side).
// Orientation mirrors GridapGmsh's own test/full_periodic.geo, which has the same point and line layout.
// Generate (from anywhere):
//   gmsh gate_family_periodic_native.geo -2 -setnumber nelemx <NX> -setnumber nelemy <NY> -o <out.msh>

If (!Exists(nelemx))
  nelemx = 4;
EndIf
If (!Exists(nelemy))
  nelemy = 4;
EndIf

xmin = -5; xmax = 5;
ymin =  0; ymax = 20;
gridsize = (xmax - xmin) / nelemx;

Point(1) = {xmin, ymin, 0, gridsize};
Point(2) = {xmax, ymin, 0, gridsize};
Point(3) = {xmax, ymax, 0, gridsize};
Point(4) = {xmin, ymax, 0, gridsize};

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

Periodic Curve {3} = {1} Translate {0, ymax - ymin, 0};
Periodic Curve {2} = {4} Translate {xmax - xmin, 0, 0};

Physical Point("boundary", 1) = {1, 2, 3, 4};
Physical Curve("periodicx", 2) = {1, 3};
Physical Curve("periodicz", 3) = {2, 4};
Physical Surface("domain", 4) = {12};
