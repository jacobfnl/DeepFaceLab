from enum import IntEnum

class FaceType(IntEnum):
    HALF = 0,
    FULL = 1,
    HEAD = 2,
    AVATAR = 3,    #centered nose only
    
    FULL_NO_ROTATION = 5,
    MARK_ONLY = 10, #no align at all, just embedded faceinfo
    
    @staticmethod
    def fromString (s):
        r = from_string_dict.get (s.lower())
        if r is None:
            raise Exception ('FaceType.fromString value error')
        return r

    @staticmethod
    def toString (face_type):
        return to_string_list[face_type]

from_string_dict = {'half_face': FaceType.HALF,
                    'full_face': FaceType.FULL,
                    'head' : FaceType.HEAD,
                    'avatar' : FaceType.AVATAR,
                    'mark_only' : FaceType.MARK_ONLY,
                    'full_face_no_rotation' : FaceType.FULL_NO_ROTATION,
                    }
to_string_list = [ 'half_face',
                   'full_face',
                   'head',
                   'avatar',
                   'mark_only',
                   'full_face_no_rotation'
                    ]
