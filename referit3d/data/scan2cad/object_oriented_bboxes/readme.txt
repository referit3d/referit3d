data structure of "object_oriented_bboxes.json"

object_oriented_bboxes = 
{
    'scene0000_00':[
                {
                    'scannet_objId': '1',                  \\ the id of the scannet object
                    'catid_cad': '03001627',               \\ the type of the shapenet model
                    'id_cad':'ffed7e95160f8edcdea0b1aceafe4876',  \\ the id of the shapenet model
                    'obj_oriented_bbox': [cx, cy, cz, lx, ly, lz, rotx, roty],
                                                           \\ the center xyz, the length xyz and
                                                           \\ the xy coordinates of the rotation vector around z axis
                    'front_point': [x, y, z]               \\ the point on center of the front face
                }
                ……
                ……
                ……
                {
                    'scannet_objI'd: '6',
                    'catid_cad': '03001627',
                    'id_cad':'21cd62313612a7a168c2f5eb1dd4dfaa',
                    'obj_oriented_bbox': [cx, cy, cz, lx, ly, lz, rotx, roty],
                    'front_point': [x, y, z]
                }
            ]
    ……
    ……
    ……
    'scene0077_00':……
    ……
}
                