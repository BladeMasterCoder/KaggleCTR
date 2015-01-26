package com.mayday.CTR;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.HashSet;

public class ClickPrediction {

	// model parameter
	static Double alpha = 0.1;  // 学习速率
	static Double beta = 1.0; 	// 自适应学习速率的参数
	static Double L1 = 1.0;
	static Double L2 = 1.0;

	static int epoch = 1;

	// 特征、散列方法
	static Integer D = 1000000;
	static Double N[] = new Double[D];
	static Double Z[] = new Double[D];
	static Double W[] = new Double[D];

	static {
		for (int i = 0; i < D; i++) {
			N[i] = 0.0;
			Z[i] = 0.0;
			W[i] = 0.0;
		}
	}

	public static void main(String[] args) {
		long startTime = System.currentTimeMillis();
		System.out.println("开始训练");
		train();
		System.out.println("训练完成，开始预测");
		test();
		// logloss();
		System.out.println("预测完成");
		long endTime = System.currentTimeMillis();
		long trainTime = endTime - startTime;
		System.out.println("用时" + trainTime / 60000 + "分钟"+ (trainTime % 60000) / 1000.0 + "秒");

	}

	public static void train() {
		String trainPath = "Data/train_rev2.csv"; // 训练文件路径
		BufferedReader br;
		String string = null;

		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(trainPath), "UTF-8"));
			string = br.readLine();
			String name[] = string.split(",");
			String value[] = null;
			HashSet<Integer> set = new HashSet<Integer>();
			for (int c = 0; c < epoch; c++) {				
				while ((string = br.readLine()) != null) {
					value = string.split(",");
					for (int i = 2; i < value.length; i++) {
						Integer hashValue = Math.abs((name[i] + "_" + value[i]).hashCode()) % D;
						set.add(hashValue);
					}
					Double p = 0.0;
					for (Integer i : set) {
						int sign = Z[i] < 0 ? -1 : 1;
						if (Math.abs(Z[i]) <= L1) {
							W[i] = 0.0;
						} else {
							W[i] = (sign * L1 - Z[i])/ ((beta + Math.sqrt(N[i])) / alpha + L2);
						}
						p += W[i];
					}
					
					// predict
					p = 1 / (1 + Math.exp(-p));
					
					// update
					Double g = p - Double.parseDouble(value[1]);
					for (Integer i : set) {
						Double sigma = (Math.sqrt(N[i] + g * g) - Math.sqrt(N[i])) / alpha;
						Z[i] += g - sigma * W[i];
						N[i] += g * g;
					}
					set.clear();
				}
				br = new BufferedReader(new InputStreamReader(new FileInputStream(trainPath), "UTF-8"));
				br.readLine();
			}
			br.close();

		} catch (UnsupportedEncodingException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

	}

	public static void test() {
		String testPath = "Data/test_rev2.csv";			 // 测试文件路径
		String submissionPath = "Data/submission.csv";	 // 结果文件路径
		BufferedReader br;
		BufferedOutputStream bos;
		String string = null;
		byte[] newLine = "\r\n".getBytes();
		int count = 0;
		try {
			bos = new BufferedOutputStream(new FileOutputStream(submissionPath));
			bos.write(("id,click").getBytes());
			bos.write(newLine);

			br = new BufferedReader(new InputStreamReader(new FileInputStream(testPath), "UTF-8"));
			string = br.readLine();
			String name[] = string.split(",");
			String value[] = null;
			HashSet<Integer> set = new HashSet<Integer>();

			while ((string = br.readLine()) != null) 
			{
				count++;
				value = string.split(",");
				for (int i = 1; i < value.length; i++) 
				{
					Integer hashValue = Math.abs((name[i] + "_" + value[i]).hashCode()) % D;
					set.add(hashValue);
				}

				Double p = 0.0;
				for (Integer i : set) {
					p += W[i];
				}
				// predict
				p = 1 / (1 + Math.exp(-p));
				String result = value[0] + "," + p;
				bos.write(result.getBytes());
				bos.write(newLine);
				set.clear();
			}
			bos.flush();
			bos.close();
			System.out.println(count);

		} catch (UnsupportedEncodingException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

	}

	public static void logloss() {
		String trainPath = "Data/train_rev2.csv"; // 训练文件路径
		BufferedReader br;
		String string = null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(
					trainPath), "UTF-8"));

			string = br.readLine();
			String name[] = string.split(",");
			String value[] = null;
			HashSet<Integer> set = new HashSet<Integer>();
			Double count = 0.0;
			Double loss = 0.0;
			while ((string = br.readLine()) != null) {
				int t = 1 + (int) (Math.random() * 5);
				if (t < 5) {
					continue;
				}
				value = string.split(",");
				for (int i = 2; i < value.length; i++) {
					Integer hashValue = Math.abs((name[i] + "_" + value[i])
							.hashCode()) % D;
					set.add(hashValue);
				}
				Double p = 0.0;
				for (Integer i : set) {
					p += W[i];
				}
				Double y = Double.parseDouble(value[1]);
				// predict
				p = 1 / (1 + Math.exp(-p));
				if (p > (1.0 - 10 * Math.exp(-15))) {
					p = 1.0 - 10 * Math.exp(-15);
				}
				if (p < 10 * Math.exp(15)) {
					p = 10 * Math.exp(15);
				}
				if (y > 0.5) {
					p = -Math.log10(p);
				} else {
					p = -Math.log10(1.0 - p);
				}
				loss += p;
				count++;
			}

			System.out.println(loss / count);
		} catch (UnsupportedEncodingException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}

}
