__kernel void brute_H(
   __global float* ve,
   __global float* gama,
   __global float* w,
   __global float* H,
   const  float z0,
   const unsigned int count_gama,
   const unsigned int count_ve
)
{
   int i_gama = get_global_id(1);
   int i_ve = get_global_id(0);
   int n;
   float temp;
   float sum;
   float gama_i = gama[i_gama];
   float ve_i = ve[i_ve];
   if(i_ve < count_ve && i_gama < count_gama){
       temp = 0;
       sum = 0;
       for (n=1; n <= 20; n++){
           temp = 1/pow(gama_i, n)/(1+(n*gama_i/(*w))*(n*gama_i/(*w)));
           if (n%2==0) temp = -temp;
           sum += temp;
       }
       sum = (ve_i*gama_i/((*w)*(*w)))*sum;
       H[i_ve*count_gama + i_gama] = z0 + sum;
   }
}
