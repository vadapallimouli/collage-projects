package Ticketsproject2.ui;

import java.awt.EventQueue;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;
import javax.swing.JLabel;
import javax.swing.JTextField;
import javax.swing.JPasswordField;
import javax.swing.JButton;
import java.awt.Font;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import javax.swing.JOptionPane;

public class LoginFrame extends JFrame {

	private static final long serialVersionUID = 1L;
	private JPanel contentPane;
	private JTextField textField;
	private JPasswordField passwordField;

	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					LoginFrame frame = new LoginFrame();
					frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	public LoginFrame() {
		setResizable(false);
		setTitle("Ticket Management System");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 1000, 700);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(null);

		JLabel lblNewLabel = new JLabel("TICKET MANAGEMENT SYSTEM");
		lblNewLabel.setFont(new Font("SansSerif", Font.BOLD, 28));
		lblNewLabel.setBounds(250, 120, 500, 40);
		contentPane.add(lblNewLabel);

		JLabel lblNewLabel_1 = new JLabel("USERNAME");
		lblNewLabel_1.setFont(new Font("SansSerif", Font.BOLD, 20));
		lblNewLabel_1.setBounds(180, 280, 140, 30);
		contentPane.add(lblNewLabel_1);

		textField = new JTextField();
		textField.setBounds(360, 280, 220, 35);
		contentPane.add(textField);
		textField.setColumns(10);

		JLabel lblNewLabel_2 = new JLabel("PASSWORD");
		lblNewLabel_2.setFont(new Font("SansSerif", Font.BOLD, 20));
		lblNewLabel_2.setBounds(180, 360, 140, 30);
		contentPane.add(lblNewLabel_2);

		passwordField = new JPasswordField();
		passwordField.setBounds(360, 360, 220, 35);
		contentPane.add(passwordField);

		JButton btnNewButton = new JButton("LOGIN");
		btnNewButton.setFont(new Font("SansSerif", Font.BOLD, 20));
		btnNewButton.setBounds(340, 460, 180, 45);
		btnNewButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				String username = textField.getText().trim();
				String password = new String(passwordField.getPassword());

				if (username.equals("admin") && password.equals("admin123")) {
					AdminDashboard admin = new AdminDashboard();
					admin.setVisible(true);
					dispose();

				} else if (username.equals("agent1") && password.equals("agent123")) {
					AgentDashboard agent = new AgentDashboard("agent1");
					agent.setVisible(true);
					dispose();

				} else if (username.equals("agent2") && password.equals("agent123")) {
					AgentDashboard agent = new AgentDashboard("agent2");
					agent.setVisible(true);
					dispose();

				} else if (username.equals("customer1") && password.equals("cust123")) {
					CustomerDashboard customer = new CustomerDashboard("customer1");
					customer.setVisible(true);
					dispose();

				} else if (username.equals("customer2") && password.equals("cust123")) {
					CustomerDashboard customer = new CustomerDashboard("customer2");
					customer.setVisible(true);
					dispose();

				} else {
					JOptionPane.showMessageDialog(null, "Invalid username or password");
				}
			}
		});
		contentPane.add(btnNewButton);
	}
}
