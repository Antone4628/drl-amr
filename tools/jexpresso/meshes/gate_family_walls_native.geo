// gate_family_walls_native.geo — the gate_family_walls.geo family with NATIVE gmsh periodicity
// on whatever pair is periodic. Same domain [-5,5]x[0,20], same transfinite quads, same tags.
// Two differences from gate_family_walls.geo, both deliberate (as in gate_family_periodic_native.geo):
//   1. Points sit at z = 0 (mesh size in the fourth slot), so GridapGmsh reports Dp = 2 and
//      Jexpresso's flatten helper does not rebuild the topology and discard the periodic glue.
//   2. A Periodic Curve constraint on each periodic pair, so the .msh carries a $Periodic section
//      and GridapGmsh glues the paired vertices in the model topology (geometry stays per side).
//   walls = 0 : doubly periodic — identical to gate_family_periodic_native.geo
//   walls = 1 : channel — bottom and top are walls, left and right natively periodic
//   walls = 2 : fully walled — no periodic pair, no constraint (identical geometry to the
//               tag-only walled mesh; a native variant exists only so one .geo serves the family)
// Tag convention (measured, DG roadmap §5.15): the periodic tag names the axis the edge RUNS
// ALONG — periodicx = bottom+top (edges along x); periodicz = left+right (edges along y).
// Generate (from anywhere):
//   gmsh gate_family_walls_native.geo -2 -setnumber nelemx <NX> -setnumber nelemy <NY> -setnumber walls <W> -o <out.msh>

If (!Exists(nelemx))
  nelemx = 4;
EndIf
If (!Exists(nelemy))
  nelemy = 4;
EndIf
If (!Exists(walls))
  walls = 1;
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

If (walls == 0)
  Periodic Curve {3} = {1} Translate {0, ymax - ymin, 0};
  Periodic Curve {2} = {4} Translate {xmax - xmin, 0, 0};
EndIf
If (walls == 1)
  Periodic Curve {2} = {4} Translate {xmax - xmin, 0, 0};
EndIf

Physical Point("boundary", 1) = {1, 2, 3, 4};
If (walls == 0)
  Physical Curve("periodicx", 2) = {1, 3};
  Physical Curve("periodicz", 3) = {2, 4};
EndIf
If (walls == 1)
  Physical Curve("wall", 2) = {1, 3};
  Physical Curve("periodicz", 3) = {2, 4};
EndIf
If (walls == 2)
  Physical Curve("wall", 2) = {1, 2, 3, 4};
EndIf
Physical Surface("domain", 4) = {12};
