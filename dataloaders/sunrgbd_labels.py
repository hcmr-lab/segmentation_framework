labels = [
	[0,		'Ignore',			[0	,	0,	0]],	
	[1,		'wall',				[98,   59, 14]],
 	[2,		'floor',			[12,   57, 226]],
 	[3,		'cabinet',			[ 83, 144,  31]],
 	[4,		'bed',				[161, 141,  54]],
 	[5,		'chair',			[165, 189, 218]],
 	[6,		'sofa',				[241,  51,  10]],
 	[7,		'table',			[ 11,  97,  48]],
 	[8,		'door',				[159, 177, 237]],
 	[9,		'window',			[33,  107,  43]],
 	[10,	'bookshelf',		[135, 248, 213]],
 	[11,	'picture',			[225,  79, 175]],
 	[12,	'counter',			[ 14, 232,   9]],
 	[13,	'blinds',			[ 91, 96, 164]],
	[14,	'desk',				[101, 42,  34]],
 	[15,	'shelves',			[ 69, 78, 232]],
 	[16,	'curtain',			[19, 180, 97]],
 	[17,	'dresser',			[44, 182, 106]],
 	[18,	'pillow',			[205, 142, 95]],
 	[19,	'mirror',			[ 42,  56,216]],
 	[20,	'floor_mat',		[186, 194,193]],
 	[21,	'clothes',			[ 34,  21,  6]],
 	[22,	'ceiling',			[119, 151, 94]],
 	[23,	'books',			[ 71, 237,  2]],
 	[24,	'fridge',			[227, 246, 31]],
 	[25,	'tv',				[170,  82, 51]],
 	[26,	'paper',			[ 48, 167, 191]],
 	[27,	'towel',			[25, 7, 159]],
 	[28,	'shower_curtain',	[135, 99, 144]],
 	[29,	'box',				[35, 63, 245]],
 	[30,	'whiteboard',		[58, 51, 121]],
 	[31,	'person',			[208, 213, 105]],
 	[32,	'night_stand',		[113, 195, 221]],
 	[33,	'toilet',			[52, 10, 95]],
 	[34,	'sink',				[136, 125, 156]],
 	[35,	'lamp',				[23, 252, 112]],
 	[36,	'bathtub',			[40, 124, 164]],
 	[37,	'bag',				[12, 65, 184]]
]


LABELS = [label[1] for label in labels]
labelid    = [label[0] for label in labels]
id2color   = {label[0]:label[2] for label in labels}
label2name = {label[0]:label[1] for label in labels}
label2id   = {label[1]:label[0] for label in labels}
id2label   = {label[0]:label[0] for label in labels}
label2color= {label[0]:label[2] for label in labels}
