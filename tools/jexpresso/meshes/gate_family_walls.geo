// gate_family_walls.geo — the 2D DG gate mesh family with a choice of walls.
// Same geometry and transfinite quads as conv_family_periodic.geo: domain [-5,5]x[0,20].
// Periodicity in this family is carried by the curve tag name alone (the reader treats
// "periodicx" / "periodicz" as periodic); a wall is the same curve under a non-periodic tag.
//   walls = 0 : doubly periodic (identical tags to conv_family_periodic.geo)
//   walls = 1 : channel — bottom and top are walls, left and right periodic
//   walls = 2 : fully walled — all four sides are walls
// Tag convention (measured, DG roadmap §5.15): the periodic tag names the axis the edge
// RUNS ALONG — periodicx = bottom+top (edges along x); periodicz = left+right (edges along y).
// Generate (from anywhere):
//   gmsh gate_family_walls.geo -2 -setnumber nelemx <NX> -setnumber nelemy <NY> -setnumber walls <W> -o <out.msh>

If (!Exists(nelemx))
  nelemx = 4;
EndIf
If (!Exists(nelemy))
  nelemy = 4;
EndIf
If (!Exists(walls))
  walls = 2;
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
