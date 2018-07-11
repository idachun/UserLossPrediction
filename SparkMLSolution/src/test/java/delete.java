import java.io.File;

public class delete {
    public static void main(String[] args){
        String filePath="";
        String filePath1="";
        for(int i=0;i<=29;i++){
            filePath = "F:\\MyDocuments\\starcor\\user_loss_1.1\\data\\play_log2\\day="+(20180401+i);
            File dir=new  File(filePath);
            for(String x:dir.list()){
                if(x.equals("minute=0000"))
                    continue;
                filePath1=filePath+"\\"+x;
                File path1=new File(filePath1);
                path1.delete();
                }
            }
        }

    }

