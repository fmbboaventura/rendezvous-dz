__kernel void vz(
    __global float* X,
    __global float* ve,
    __global float* gama,
    __global float* w,
    __global float* I,
    __global float* jn,
    __global float* H,
    __global float* vz,
    const unsigned int count_gama,
    const unsigned int count_ve,
    const unsigned int count_X
)
{
    int t = 1 + get_global_id(0);
    int X_i = 1 + get_global_id(1);
    int i_X = get_global_id(1);
    int offset = get_global_offset(0);

    int i_gama;
    int i_ve;
    int n;
    int count_T = get_global_size(0);

    float gama_i;
    float ve_i;
    float temp;

    float H_i;
    float I_i;

    if(t <= (count_T + offset) && i_X <= count_X)
    {
        for(i_ve = 0; i_ve < count_ve; i_ve++)
        {
            for(i_gama = 0; i_gama < count_gama; i_gama++)
            {
                H_i = H[i_ve*count_gama + i_gama];
                I_i = I[i_X*count_ve*count_gama + i_ve*count_gama + i_gama];
                temp = 0;
                for (n=1; n <= 20; n++){
                    temp += jn[i_X*count_ve*count_gama*20 +
                        i_ve*count_gama*20 +
                        i_gama*20 + (n-1)] *
                        (-n*gama_i*exp(-n*gama_i*t));
                }

                // vz[
                //     (t-offset-1)*count_X*count_ve*count_gama +
                //     i_X*count_ve*count_gama +
                //     i_ve*count_gama + i_gama
                // ] = t;
                //(*w)*(-H_i*sin((*w)*t) + I_i*cos((*w)*t)) - temp;
                vz[
                    i_X*count_ve*count_gama*count_T +
                    i_ve*count_gama*count_T +
                    i_gama*count_T + (t-offset-1)
                 ] = (*w)*(-H_i*sin((*w)*t) + I_i*cos((*w)*t)) - temp;
            }
        }
    }
}
