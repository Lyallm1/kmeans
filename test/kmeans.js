var kmeans = require('..');

var data = [[-0.0666, 3.9556], [4.76, 3.3532], [-3.1439, 11.9022], [8.8348, 11.7216], [-0.5141, -1.0318], [-0.8734, -0.0638], [1.8586, 0.1617], [2.5949, 1.7523], [-9.6961, 6.0031], [-8.9181, -5.1248], [7.1932, 12.0728], [0.0514, 0.2759], [6.9711, 13.1064], [-10.9973, 7.7045], [6.3518, 13.8093], [10.9279, 5.1009], [10.4232, -10.2709], [-3.2731, 1.5736], [-10.7068, 1.1708], [10.0436, 12.3809], [-7.3017, -8.8664], [14.3335, -0.4248], [-1.2148, 1.2219], [2.4937, 11.4931], [11.9281, -0.3832], [0.9821, 4.2688], [2.37, -9.7278], [14.6577, -2.3196], [12.0509, -1.1881], [0.4072, -1.5841], [-12.4332, 9.5092], [10.0304, -3.3084], [-3.2987, 2.6475], [-9.183, -6.732], [1.4777, -0.1288], [4.9672, 2.2459], [3.378, 11.7724], [-1.0655, -13.701], [2.1222, -14.0421], [-11.9715, 8.6394], [4.2109, 2.648], [-14.0372, 4.3083], [-3.925, 11.4749], [8.1935, 8.5127], [-0.4264, -5.4829], [-0.0024, -0.0029], [15.7082, 0.5469], [11.2737, -10.5358], [3.6512, -4.2767], [-15.1431, -2.9218], [9.0929, -12.6038], [-11.1781, -6.6943], [12.7945, 1.2342], [-6.133, 9.8605], [-12.6114, 7.7679], [-9.9677, -7.7301], [-4.6634, 12.9643], [1.5693, -12.5789], [5.7557, 9.6854], [6.6642, -8.3058], [-6.8809, -13.7961], [2.4091, 4.5483], [-2.5279, 1.7826], [-2.4686, -1.0076], [-1.627, 3.8966], [-11.2086, -10.4047], [0.4449, 0.9771], [3.1429, -11.6762], [-5.6467, -9.8625], [-0.1443, 1.1105], [11.4457, 7.1273], [-2.0661, -0.8904], [10.4427, 6.7074], [13.3232, 3.8624], [-1.3954, 10.7406], [-11.0965, 7.4209], [-7.4238, -10.2261], [12.7514, -5.1493], [4.5682, 14.3597], [0.0111, -0.2019], [4.4757, -14.9786], [5.5468, 12.642], [2.9031, 0.9922], [13.5853, 5.7207], [0.6769, 1.2887], [6.0625, 14.6495], [0.5118, -3.8389], [2.7598, -2.707], [3.0966, 1.3363], [-2.753, 0.9607], [9.6514, 5.0441], [-0.4091, 1.3676], [-9.6076, 8.0624], [-8.7501, -11.9083], [10.0703, -6.6664], [-2.775, 4.9923], [1.7828, 2.6811], [-8.3187, -9.0107], [7.869, -11.9879], [4.4372, -10.9654], [-13.9686, -3.7593], [4.7448, -0.5838], [1.523, -1.1613], [-2.7016, -12.164], [-0.101, -1.0998], [-4.298, -2.6621], [-0.4873, -4.2047], [11.2547, 9.8242], [6.1379, 12.4645], [-9.5691, 7.0038], [9.8499, 8.2545], [-5.5557, 9.7861], [1.9109, 0.5991], [2.349, -1.0844], [7.8578, -12.2351], [3.0412, 0.0989], [-0.0646, -0.1736], [-12.9012, 1.3657], [4.2466, -14.8209], [11.5388, 2.4394], [3.929, -3.9739], [0.9316, 3.905], [4.5496, 1.1866], [4.372, -2.7704], [11.455, 4.7257], [0.1667, 3.0992], [3.9864, 2.1817], [4.305, -1.6716], [7.562, -9.0314], [-14.2041, -3.4945], [-1.8313, 5.4946], [-0.2621, 2.7548], [1.2749, -5.5176], [-3.2882, -0.9906], [13.4358, 7.2966], [-0.2678, 12.9615], [-12.3131, -6.7527], [-0.6749, 12.4802], [-2.8935, -0.1917], [1.2031, 13.7469], [2.7249, -12.0991], [6.8316, 13.5634], [-12.2154, 8.1218], [-11.2229, 2.5678], [1.2659, 3.8962], [0.0282, -13.0606], [4.6974, 12.4882], [-11.1419, 4.5603], [6.2891, 7.8661], [0.6523, -0.2886], [-9.0433, -8.0838], [-1.5765, 4.7914], [6.2527, 13.5528], [-14.2127, -5.392], [2.622, 1.23], [15.2967, 1.124], [0.1564, -10.5582], [-0.0184, -1.1081], [1.3052, 2.704], [-0.4534, -4.2774], [8.2172, -9.4401], [3.2584, 11.923], [0.5783, 12.0416], [-10.394, 7.1666], [6.2146, -13.9815], [8.274, -7.4318], [5.8541, -8.6234], [-10.5098, 10.1849], [11.9771, 4.8152], [-0.1449, -0.1554], [0.142, 0.246], [-14.4965, 1.4054], [0.1092, -0.7848], [-0.9279, -0.5001], [12.741, 7.6121], [4.9104, 3.154], [11.4821, 4.2069], [-4.823, 10.66], [11.6895, -3.2821], [-11.1159, 9.4047], [9.6862, 7.0255], [4.2709, -11.0134], [0.6158, 0.1453], [0.1495, 0.3122], [-13.0746, 1.2675], [11.5982, -0.8388], [-12.8156, -2.8367], [8.8404, -11.7197], [1.9458, -10.7101], [-4.7165, -2.0864], [0.9491, -2.7463], [2.8534, -15.6712], [-10.9901, 2.373], [0.5304, 0.1503], [-0.9909, 10.8084], [-8.2816, -6.3139], [0.4797, 1.9009], [-10.8144, 4.861], [-0.3332, -10.806], [12.2777, -9.5046], [10.8568, 3.2378], [-13.6074, -1.8163], [-5.0088, -14.0002], [-5.4378, 10.2545], [-14.7814, -3.7871], [4.799, 2.5422], [1.8227, -2.1769], [-8.4743, 6.0638], [3.2532, 13.2978], [0.1051, 11.8516], [4.8026, -9.8529], [1.8074, -3.4619], [6.1632, 14.2468], [12.1793, -2.0449], [-11.8355, -8.555], [0.8006, -2.2425], [-4.8806, 10.4182], [11.7955, -0.6237], [9.7098, -4.5823], [-11.3679, -7.2694], [-0.4442, 12.2027], [-10.0847, 8.4332], [-0.2568, -12.0907], [4.5121, -0.4623], [-11.1321, -0.2562], [-7.8491, -11.8705], [8.8087, 11.5666], [1.2531, 12.7417], [1.296, 10.6887], [0.178, 4.1553], [-10.2494, 7.4416], [5.2538, -1.7206], [-1.5211, 12.4555], [-2.8232, -3.3191], [-10.55, -10.1068], [0.525, 1.0544], [-9.5425, 2.991], [-7.8366, -10.869], [10.4709, 10.306], [2.9398, -4.0552], [11.4418, 11.167], [3.8228, 10.7863], [8.819, -8.2799], [15.0613, -5.0608], [-4.5679, -3.7592], [-7.3167, -10.3306], [-1.136, 0.6528], [5.9679, 9.9295], [-5.1458, 10.8432], [3.3021, 0.6736], [-3.3911, -1.4024], [-0.6543, 2.1858], [1.9018, 1.6944], [12.3353, 5.2557], [-12.6925, 9.0565], [1.5401, -2.8375], [-12.5521, -5.8312], [-8.6292, 7.6424], [0.0011, -0.7609], [3.2544, 4.6448], [-8.9116, 8.4947], [2.2136, 11.4343], [13.7698, -7.6187], [-2.125, 15.5614], [14.7314, -2.8995], [-2.9476, 10.2836], [10.3428, -7.124], [-4.0868, -9.6519], [-13.3819, 0.1771], [2.412, 3.4523], [-9.4296, -11.8803], [5.8335, 12.6889], [-1.2018, 1.3509], [-13.8312, 0.1857], [-1.9669, -11.8927], [-5.2092, -15.1251], [12.8108, -8.4382], [-9.9785, -2.8565], [2.9082, -14.2871], [-10.1149, -10.645], [-1.314, -14.0368], [0.6984, 1.3434], [-3.3646, 9.7148], [-0.6432, -3.3273], [-9.9093, 3.7181], [-9.6247, -7.4199], [-10.6138, 3.0472], [-13.2729, -7.2328], [3.4595, 1.7802], [5.9987, -13.601], [-12.179, -2.9048], [1.3147, -0.3124], [4.9982, -9.1045], [-5.554, -13.3016], [-1.9886, 3.3253], [5.7266, -8.4147], [-10.3248, -1.2052], [-13.2403, 8.1238], [-6.2474, 11.419], [-1.2548, -14.8794]];
var centers = [[-0.0666, 3.9556], [4.76, 3.3532]];
var clusterID = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0];

describe('K-means test', function () {

    it('main test', function () {
        var ans = kmeans(data, centers);
        for (var i = 0, l = clusterID.length; i < l; i++) {
            ans[i].should.equal(clusterID[i]);
        }
    });
});