function INDEX=BuildChunk(N,n)
        if (N==0)|(N==n),INDEX=[1,n]; 
        else
       
        k=2;INDEX=[1,N];
        flag=1;
        while flag
            INDEX(k,:)=[INDEX(k-1,2)+1,INDEX(k-1,2)+N];
            if INDEX(k,2)>=n,INDEX(k,2)=n;flag=0;end
            k=k+1;
        end
        end