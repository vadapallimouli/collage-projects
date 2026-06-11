package Ticketsproject2.ui;

import java.awt.EventQueue;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.table.DefaultTableModel;

import Ticketsproject2.database.DBConnection;

public class CustomerDashboard extends JFrame {

	private static final long serialVersionUID = 1L;

	private JTextField textField;
	private JTextField textField_1;
	private JTable table;
	private JTextArea textArea;

	private JComboBox<String> comboBox;
	private JComboBox<String> comboBox_1;

	private JLabel lblTotalCount;
	private JLabel lblOpenCount;
	private JLabel lblProgressCount;
	private JLabel lblClosedCount;

	private String customerUsername = "customer1";
	private int selectedTicketId = -1;

	public static void main(String[] args) {
		EventQueue.invokeLater(() -> {
			try {
				CustomerDashboard frame = new CustomerDashboard("customer1");
				frame.setVisible(true);
			} catch (Exception e) {
				e.printStackTrace();
			}
		});
	}

	public CustomerDashboard() {
		this("customer1");
	}

	public CustomerDashboard(String customerUsername) {
		this.customerUsername = customerUsername;

		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 1000, 700);
		getContentPane().setLayout(null);

		JLabel lblTitle = new JLabel("CUSTOMER DASHBOARD");
		lblTitle.setBounds(20, 15, 220, 30);
		getContentPane().add(lblTitle);

		JLabel lblCustomerName = new JLabel("Logged in as: " + this.customerUsername);
		lblCustomerName.setBounds(240, 15, 220, 30);
		getContentPane().add(lblCustomerName);

		JButton btnLogout = new JButton("Logout");
		btnLogout.setBounds(860, 15, 100, 30);
		getContentPane().add(btnLogout);

		JLabel lblTotal = new JLabel("Total Tickets");
		lblTotal.setBounds(60, 70, 100, 20);
		getContentPane().add(lblTotal);

		JLabel lblOpen = new JLabel("Open Tickets");
		lblOpen.setBounds(280, 70, 100, 20);
		getContentPane().add(lblOpen);

		JLabel lblProgress = new JLabel("In Progress");
		lblProgress.setBounds(500, 70, 100, 20);
		getContentPane().add(lblProgress);

		JLabel lblClosed = new JLabel("Closed Tickets");
		lblClosed.setBounds(720, 70, 100, 20);
		getContentPane().add(lblClosed);

		lblTotalCount = new JLabel("0");
		lblTotalCount.setBounds(100, 100, 40, 20);
		getContentPane().add(lblTotalCount);

		lblOpenCount = new JLabel("0");
		lblOpenCount.setBounds(320, 100, 40, 20);
		getContentPane().add(lblOpenCount);

		lblProgressCount = new JLabel("0");
		lblProgressCount.setBounds(540, 100, 40, 20);
		getContentPane().add(lblProgressCount);

		lblClosedCount = new JLabel("0");
		lblClosedCount.setBounds(760, 100, 40, 20);
		getContentPane().add(lblClosedCount);

		JLabel lblName = new JLabel("Name:");
		lblName.setBounds(20, 150, 90, 20);
		getContentPane().add(lblName);

		textField = new JTextField();
		textField.setBounds(110, 147, 160, 28);
		textField.setColumns(10);
		getContentPane().add(textField);

		JLabel lblPriority = new JLabel("Priority:");
		lblPriority.setBounds(300, 150, 60, 20);
		getContentPane().add(lblPriority);

		comboBox = new JComboBox<>();
		comboBox.setModel(new DefaultComboBoxModel<>(new String[] {
				"Low", "Medium", "High", "Critical"
		}));
		comboBox.setBounds(360, 147, 140, 28);
		getContentPane().add(comboBox);

		JLabel lblAssignAgent = new JLabel("Assign Agent:");
		lblAssignAgent.setBounds(530, 150, 100, 20);
		getContentPane().add(lblAssignAgent);

		comboBox_1 = new JComboBox<>();
		comboBox_1.setModel(new DefaultComboBoxModel<>(new String[] {
				"agent1", "agent2"
		}));
		comboBox_1.setBounds(630, 147, 140, 28);
		getContentPane().add(comboBox_1);

		JLabel lblIssue = new JLabel("Issue:");
		lblIssue.setBounds(20, 195, 60, 20);
		getContentPane().add(lblIssue);

		textField_1 = new JTextField();
		textField_1.setBounds(80, 192, 300, 28);
		textField_1.setColumns(10);
		getContentPane().add(textField_1);

		JLabel lblProblem = new JLabel("Problem:");
		lblProblem.setBounds(20, 240, 70, 20);
		getContentPane().add(lblProblem);

		JScrollPane scrollPane_1 = new JScrollPane();
		scrollPane_1.setBounds(90, 240, 680, 100);
		getContentPane().add(scrollPane_1);

		textArea = new JTextArea();
		scrollPane_1.setViewportView(textArea);

		JButton btnCreateTicket = new JButton("Create Ticket");
		btnCreateTicket.setBounds(800, 260, 140, 35);
		getContentPane().add(btnCreateTicket);

		JButton btnModifyTicket = new JButton("Modify");
		btnModifyTicket.setBounds(800, 310, 140, 35);
		getContentPane().add(btnModifyTicket);

		JScrollPane scrollPane = new JScrollPane();
		scrollPane.setBounds(20, 380, 940, 220);
		getContentPane().add(scrollPane);

		table = new JTable();
		table.setModel(new DefaultTableModel(
			new Object[][] {
			},
			new String[] {
				"ID", "Name", "Issue", "Priority", "Assigned Agent", "Status"
			}
		) {
			private static final long serialVersionUID = 1L;

			@Override
			public boolean isCellEditable(int row, int column) {
				return false;
			}
		});
		scrollPane.setViewportView(table);

		btnLogout.addActionListener(e -> dispose());
		btnCreateTicket.addActionListener(e -> createTicket());
		btnModifyTicket.addActionListener(e -> modifyTicket());

		table.getSelectionModel().addListSelectionListener(e -> {
			if (!e.getValueIsAdjusting()) {
				fillFormFromTable();
			}
		});

		loadCustomerTickets();
		loadStats();
	}

	private String getCurrentDisplayName() {
		String name = textField.getText().trim();
		return name.isEmpty() ? customerUsername : name;
	}

	private void createTicket() {
		String displayName = getCurrentDisplayName();
		String issueTitle = textField_1.getText().trim();
		String priority = comboBox.getSelectedItem().toString();
		String assignedAgent = comboBox_1.getSelectedItem().toString();
		String problem = textArea.getText().trim();

		if (displayName.isEmpty()) {
			JOptionPane.showMessageDialog(this, "Enter name");
			return;
		}

		if (issueTitle.isEmpty()) {
			JOptionPane.showMessageDialog(this, "Enter issue");
			return;
		}

		if (problem.isEmpty()) {
			JOptionPane.showMessageDialog(this, "Enter problem description");
			return;
		}

		String fullIssue = issueTitle + " - " + problem;

		String sql = "INSERT INTO tickets (customerName, displayName, issue, priority, category, assignedAgent, status, createdAt) "
				+ "VALUES (?, ?, ?, ?, ?, ?, 'Open', NOW())";

		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql, PreparedStatement.RETURN_GENERATED_KEYS)) {

			pst.setString(1, customerUsername);
			pst.setString(2, displayName);
			pst.setString(3, fullIssue);
			pst.setString(4, priority);
			pst.setString(5, "Support");
			pst.setString(6, assignedAgent);

			pst.executeUpdate();

			try (ResultSet rs = pst.getGeneratedKeys()) {
				if (rs.next()) {
					int generatedId = rs.getInt(1);
					JOptionPane.showMessageDialog(this, "Ticket created successfully. Ticket ID: " + generatedId);
				} else {
					JOptionPane.showMessageDialog(this, "Ticket created successfully.");
				}
			}

			clearForm();
			loadCustomerTickets();
			loadStats();

		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Create ticket error: " + e.getMessage());
		}
	}

	private void fillFormFromTable() {
		int row = table.getSelectedRow();

		if (row >= 0) {
			String status = table.getValueAt(row, 5).toString();

			if (!"Open".equalsIgnoreCase(status)) {
				selectedTicketId = -1;
				JOptionPane.showMessageDialog(this, "Only open tickets can be modified");
				return;
			}

			selectedTicketId = Integer.parseInt(table.getValueAt(row, 0).toString());

			String displayName = table.getValueAt(row, 1).toString();
			String issueText = table.getValueAt(row, 2).toString();
			String priority = table.getValueAt(row, 3).toString();
			String assignedAgent = table.getValueAt(row, 4).toString();

			textField.setText(displayName);
			comboBox.setSelectedItem(priority);
			comboBox_1.setSelectedItem(assignedAgent);

			if (issueText.contains(" - ")) {
				String[] issueParts = issueText.split(" - ", 2);
				textField_1.setText(issueParts[0].trim());
				textArea.setText(issueParts[1].trim());
			} else {
				textField_1.setText(issueText);
				textArea.setText("");
			}
		}
	}

	private void modifyTicket() {
		if (selectedTicketId == -1) {
			JOptionPane.showMessageDialog(this, "Select an open ticket first");
			return;
		}

		String displayName = getCurrentDisplayName();
		String issueTitle = textField_1.getText().trim();
		String priority = comboBox.getSelectedItem().toString();
		String assignedAgent = comboBox_1.getSelectedItem().toString();
		String problem = textArea.getText().trim();

		if (displayName.isEmpty()) {
			JOptionPane.showMessageDialog(this, "Enter name");
			return;
		}

		if (issueTitle.isEmpty()) {
			JOptionPane.showMessageDialog(this, "Enter issue");
			return;
		}

		if (problem.isEmpty()) {
			JOptionPane.showMessageDialog(this, "Enter problem description");
			return;
		}

		String fullIssue = issueTitle + " - " + problem;

		String sql = "UPDATE tickets SET displayName = ?, issue = ?, priority = ?, assignedAgent = ? "
				+ "WHERE id = ? AND customerName = ? AND status = 'Open'";

		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql)) {

			pst.setString(1, displayName);
			pst.setString(2, fullIssue);
			pst.setString(3, priority);
			pst.setString(4, assignedAgent);
			pst.setInt(5, selectedTicketId);
			pst.setString(6, customerUsername);

			int updated = pst.executeUpdate();

			if (updated > 0) {
				JOptionPane.showMessageDialog(this, "Ticket modified successfully");
				clearForm();
				loadCustomerTickets();
				loadStats();
			} else {
				JOptionPane.showMessageDialog(this, "Only open tickets can be modified");
			}

		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Modify ticket error: " + e.getMessage());
		}
	}

	private void clearForm() {
		selectedTicketId = -1;
		textField.setText("");
		textField_1.setText("");
		textArea.setText("");
		comboBox.setSelectedIndex(0);
		comboBox_1.setSelectedIndex(0);
	}

	private void loadCustomerTickets() {
		DefaultTableModel model = (DefaultTableModel) table.getModel();
		model.setRowCount(0);

		String sql = "SELECT id, displayName, issue, priority, assignedAgent, status "
				+ "FROM tickets WHERE customerName = ? ORDER BY id DESC";

		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql)) {

			pst.setString(1, customerUsername);

			try (ResultSet rs = pst.executeQuery()) {
				while (rs.next()) {
					model.addRow(new Object[] {
						rs.getInt("id"),
						rs.getString("displayName"),
						rs.getString("issue"),
						rs.getString("priority"),
						rs.getString("assignedAgent"),
						rs.getString("status")
					});
				}
			}
		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Error loading tickets: " + e.getMessage());
		}
	}

	private void loadStats() {
		lblTotalCount.setText(String.valueOf(getCount(
				"SELECT COUNT(*) FROM tickets WHERE customerName = ?")));

		lblOpenCount.setText(String.valueOf(getCount(
				"SELECT COUNT(*) FROM tickets WHERE customerName = ? AND status = 'Open'")));

		lblProgressCount.setText(String.valueOf(getCount(
				"SELECT COUNT(*) FROM tickets WHERE customerName = ? AND status = 'In Progress'")));

		lblClosedCount.setText(String.valueOf(getCount(
				"SELECT COUNT(*) FROM tickets WHERE customerName = ? AND status = 'Closed'")));
	}

	private int getCount(String sql) {
		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql)) {

			pst.setString(1, customerUsername);

			try (ResultSet rs = pst.executeQuery()) {
				if (rs.next()) {
					return rs.getInt(1);
				}
			}
		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Stats error: " + e.getMessage());
		}
		return 0;
	}
}
