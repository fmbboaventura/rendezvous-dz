__kernel void brute_I(
   __global float* X,
   __global float* ve,
   __global float* gama,
   __global float* w,
   __global float* I,
   __global float* jn,
   const  float vz0,
   const unsigned int count_gama,
   const unsigned int count_ve,
   const unsigned int count_X
)
{
   int i_gama = get_global_id(2);
   int i_ve = get_global_id(1);
   int i_X = get_global_id(0);
   int n;
   float temp;
   float gama_i = gama[i_gama];
   float ve_i = ve[i_ve];
   float X_i = X[i_X];
   if(i_X < count_X && i_ve < count_ve && i_gama < count_gama){
       temp = 0;
       for (n=1; n <= 20; n++){
           temp += jn[i_X*count_ve*count_gama*20 +
               i_ve*count_gama*20 +
               i_gama*20 + (n-1)]/n;
       }
       I[i_X*count_ve*count_gama +
           i_ve*count_gama +
           i_gama] = (vz0 - ve_i*log((X_i-1)/X_i))/(*w) + temp;
   }
}
